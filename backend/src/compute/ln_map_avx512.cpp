// compute/ln_map_avx512.cpp

#include "ln_map.hpp"

#include "map_kernel_avx512.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#  include <immintrin.h>
#endif

namespace fsd::compute {

#if defined(__AVX512F__)
namespace {

constexpr double TAU = 6.283185307179586;
constexpr double LN_FOUR = 1.3862943611198906;

inline __m512d abs_pd(__m512d v) {
    const __m512d sign = _mm512_set1_pd(-0.0);
    return _mm512_andnot_pd(sign, v);
}

inline __mmask8 lane_mask(int base, int count) {
    int mask = 0;
    for (int k = 0; k < 8; ++k) {
        if (base + k < count) mask |= (1 << k);
    }
    return static_cast<__mmask8>(mask);
}

inline void step_variant(
    int variant,
    __m512d zre,
    __m512d zim,
    __m512d zre2,
    __m512d zim2,
    __m512d cre,
    __m512d cim,
    __m512d& nre,
    __m512d& nim
) {
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d sq_re = _mm512_sub_pd(zre2, zim2);
    const __m512d sq_im = _mm512_mul_pd(_mm512_mul_pd(two, zre), zim);

    switch (variant) {
        case 1:
            nre = _mm512_add_pd(sq_re, cre);
            nim = _mm512_sub_pd(cim, sq_im);
            break;
        case 2:
            nre = _mm512_add_pd(sq_re, cre);
            nim = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(two, abs_pd(zre)), abs_pd(zim)), cim);
            break;
        case 3:
            nre = _mm512_add_pd(sq_re, cre);
            nim = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(two, zre), abs_pd(zim)), cim);
            break;
        case 4:
            nre = _mm512_add_pd(sq_re, cre);
            nim = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(two, abs_pd(zre)), _mm512_sub_pd(zero, zim)), cim);
            break;
        case 5:
            nre = _mm512_add_pd(abs_pd(sq_re), cre);
            nim = _mm512_add_pd(sq_im, cim);
            break;
        case 6:
            nre = _mm512_add_pd(abs_pd(sq_re), cre);
            nim = _mm512_sub_pd(cim, sq_im);
            break;
        case 7:
            nre = _mm512_add_pd(abs_pd(sq_re), cre);
            nim = _mm512_add_pd(abs_pd(sq_im), cim);
            break;
        case 8:
            nre = _mm512_add_pd(abs_pd(sq_re), cre);
            nim = _mm512_add_pd(_mm512_mul_pd(_mm512_mul_pd(two, zre), abs_pd(zim)), cim);
            break;
        case 9:
            nre = _mm512_add_pd(abs_pd(sq_re), cre);
            nim = _mm512_sub_pd(cim, _mm512_mul_pd(_mm512_mul_pd(two, abs_pd(zre)), zim));
            break;
        default:
            nre = _mm512_add_pd(sq_re, cre);
            nim = _mm512_add_pd(sq_im, cim);
            break;
    }
}

void ensure_out(const LnMapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height_t || out.cols != p.width_s || out.type() != CV_8UC3) {
        out.create(p.height_t, p.width_s, CV_8UC3);
    }
}

std::pair<int, int> clamp_rows(const LnMapParams& p, int row_start, int row_count) {
    const int start = std::max(0, std::min(row_start, p.height_t));
    const int end = std::max(start, std::min(p.height_t, start + std::max(0, row_count)));
    return {start, end};
}

struct TrigColumns {
    std::vector<double> cos_col;
    std::vector<double> sin_col;
};

TrigColumns make_trig_columns(int s) {
    TrigColumns cols;
    cols.cos_col.resize(static_cast<size_t>(s));
    cols.sin_col.resize(static_cast<size_t>(s));
    for (int x = 0; x < s; x++) {
        const double th = TAU * static_cast<double>(x) / static_cast<double>(s);
        cols.cos_col[static_cast<size_t>(x)] = std::cos(th);
        cols.sin_col[static_cast<size_t>(x)] = std::sin(th);
    }
    return cols;
}

void render_rows_impl(
    const LnMapParams& p,
    cv::Mat& out,
    int row_start,
    int row_end,
    bool threaded,
    const LnMapProgress& on_row_done
) {
    const int s = p.width_s;
    const TrigColumns cols = make_trig_columns(s);
    const int variant = static_cast<int>(p.variant);
    const __m512d vbail2 = _mm512_set1_pd(p.bailout_sq);
    const __m512d vzero = _mm512_setzero_pd();
    const __m512d vjre = _mm512_set1_pd(p.julia_re);
    const __m512d vjim = _mm512_set1_pd(p.julia_im);
    std::atomic<int> rows_done{0};

    auto render_row = [&](int row) {
        uint8_t* rowp = out.ptr<uint8_t>(row);
        const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(s);
        const double r_mag = std::exp(k);

        for (int col = 0; col < s; col += 8) {
            alignas(64) double pre_arr[8] = {};
            alignas(64) double pim_arr[8] = {};
            for (int lane = 0; lane < 8 && col + lane < s; ++lane) {
                const size_t idx = static_cast<size_t>(col + lane);
                pre_arr[lane] = p.center_re + r_mag * cols.cos_col[idx];
                pim_arr[lane] = p.center_im + r_mag * cols.sin_col[idx];
            }

            const __mmask8 valid = lane_mask(col, s);
            const __m512d vpre = _mm512_load_pd(pre_arr);
            const __m512d vpim = _mm512_load_pd(pim_arr);
            __m512d zre = p.julia ? vpre : vzero;
            __m512d zim = p.julia ? vpim : vzero;
            const __m512d cre = p.julia ? vjre : vpre;
            const __m512d cim = p.julia ? vjim : vpim;
            __m512d zre2 = _mm512_mul_pd(zre, zre);
            __m512d zim2 = _mm512_mul_pd(zim, zim);

            alignas(64) int iter_arr[8];
            for (int lane = 0; lane < 8; ++lane) iter_arr[lane] = p.iterations;

            __mmask8 active = valid;
            for (int iter = 0; iter < p.iterations && active; ++iter) {
                __m512d nre, nim;
                step_variant(variant, zre, zim, zre2, zim2, cre, cim, nre, nim);
                const __m512d nre2 = _mm512_mul_pd(nre, nre);
                const __m512d nim2 = _mm512_mul_pd(nim, nim);
                const __m512d norm2 = _mm512_add_pd(nre2, nim2);
                const __mmask8 escaped =
                    active & (_mm512_cmp_pd_mask(norm2, vbail2, _CMP_GT_OQ) |
                              _mm512_cmp_pd_mask(norm2, norm2, _CMP_UNORD_Q));
                if (escaped) {
                    for (int lane = 0; lane < 8; ++lane) {
                        if (escaped & (1 << lane)) iter_arr[lane] = iter;
                    }
                    active &= static_cast<__mmask8>(~escaped);
                }
                zre = _mm512_mask_mov_pd(zre, active, nre);
                zim = _mm512_mask_mov_pd(zim, active, nim);
                zre2 = _mm512_mask_mov_pd(zre2, active, nre2);
                zim2 = _mm512_mask_mov_pd(zim2, active, nim2);
            }

            for (int lane = 0; lane < 8 && col + lane < s; ++lane) {
                uint8_t* px = rowp + 3 * (col + lane);
                colorize_escape_bgr(iter_arr[lane], p.iterations, p.colormap, 0.0, false, px[0], px[1], px[2]);
            }
        }

        if (on_row_done) {
            const int done = rows_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done == row_end - row_start || (done % 16) == 0) on_row_done(done);
        }
    };

    if (threaded) {
        const int thread_count = default_render_threads();
        #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 4)
        for (int row = row_start; row < row_end; ++row) {
            render_row(row);
        }
    } else {
        for (int row = row_start; row < row_end; ++row) {
            render_row(row);
        }
    }
}

} // namespace
#endif

LnMapStats render_ln_map_avx512_rows(const LnMapParams& p, cv::Mat& out, int row_start, int row_count, const LnMapProgress& on_row_done) {
    if (!avx512_available() || !ln_map_variant_supported_by_simd(p.variant)) {
        return render_ln_map_openmp_rows(p, out, row_start, row_count, on_row_done);
    }

#if defined(__AVX512F__)
    ensure_out(p, out);
    const auto [start, end] = clamp_rows(p, row_start, row_count);
    const auto t0 = std::chrono::steady_clock::now();
    render_rows_impl(p, out, start, end, false, on_row_done);
    const auto t1 = std::chrono::steady_clock::now();

    LnMapStats stats;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.pixel_count = p.width_s * (end - start);
    stats.engine_used = "avx512";
    stats.scalar_used = "fp64";
    return stats;
#else
    return render_ln_map_openmp_rows(p, out, row_start, row_count, on_row_done);
#endif
}

LnMapStats render_ln_map_avx512(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    if (!avx512_available() || !ln_map_variant_supported_by_simd(p.variant)) {
        return render_ln_map_openmp(p, out, on_row_done);
    }

#if defined(__AVX512F__)
    ensure_out(p, out);
    const auto t0 = std::chrono::steady_clock::now();
    render_rows_impl(p, out, 0, p.height_t, true, on_row_done);
    const auto t1 = std::chrono::steady_clock::now();

    LnMapStats stats;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.pixel_count = p.width_s * p.height_t;
    stats.engine_used = "avx512";
    stats.scalar_used = "fp64";
    return stats;
#else
    return render_ln_map_openmp(p, out, on_row_done);
#endif
}

} // namespace fsd::compute
