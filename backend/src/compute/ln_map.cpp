// compute/ln_map.cpp

#include "ln_map.hpp"

#include "engine_select.hpp"
#include "escape_time.hpp"
#include "map_kernel_avx512.hpp"
#include "parallel.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/ln_map.cuh"
#  define USE_CUDA_LN_MAP 1
#else
#  define USE_CUDA_LN_MAP 0
#endif

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

namespace fsd::compute {
namespace {

constexpr double TAU = 6.283185307179586;
constexpr double LN_FOUR = 1.3862943611198906;

template <Variant V>
void render_ln_variant_openmp(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    const Cx<double> c_julia{p.julia_re, p.julia_im};
    const int s = p.width_s;
    const int t = p.height_t;
    const int thread_count = default_render_threads();
    std::vector<double> cos_col(static_cast<size_t>(s));
    std::vector<double> sin_col(static_cast<size_t>(s));
    for (int x = 0; x < s; x++) {
        const double th = TAU * static_cast<double>(x) / static_cast<double>(s);
        cos_col[static_cast<size_t>(x)] = std::cos(th);
        sin_col[static_cast<size_t>(x)] = std::sin(th);
    }

    std::atomic<int> rows_done{0};
    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp for schedule(dynamic, 8)
        for (int row = 0; row < t; row++) {
            uint8_t* rowp = out.ptr<uint8_t>(row);
            const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(s);
            const double r_mag = std::exp(k);
            for (int x = 0; x < s; x++) {
                const double pre = p.center_re + r_mag * cos_col[static_cast<size_t>(x)];
                const double pim = p.center_im + r_mag * sin_col[static_cast<size_t>(x)];
                Cx<double> z0, c;
                if (p.julia) {
                    z0 = {pre, pim};
                    c = c_julia;
                } else {
                    z0 = {0.0, 0.0};
                    c = {pre, pim};
                }
                const IterResult ir = iterate_masked<
                    IterResultField::Iter | IterResultField::Escaped,
                    V, double>(z0, c, p.iterations, p.bailout, p.bailout_sq);
                uint8_t* px = rowp + 3 * x;
                const int it = ir.escaped ? ir.iter : p.iterations;
                colorize_escape_bgr(it, p.iterations, p.colormap, 0.0, false, px[0], px[1], px[2]);
            }
            if (on_row_done) {
                const int done = rows_done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done == t || (done % 16) == 0) on_row_done(done);
            }
        }
    }
}

void dispatch_openmp(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    switch (p.variant) {
        case Variant::Mandelbrot: render_ln_variant_openmp<Variant::Mandelbrot>(p, out, on_row_done); break;
        case Variant::Tri:        render_ln_variant_openmp<Variant::Tri>       (p, out, on_row_done); break;
        case Variant::Boat:       render_ln_variant_openmp<Variant::Boat>      (p, out, on_row_done); break;
        case Variant::Duck:       render_ln_variant_openmp<Variant::Duck>      (p, out, on_row_done); break;
        case Variant::Bell:       render_ln_variant_openmp<Variant::Bell>      (p, out, on_row_done); break;
        case Variant::Fish:       render_ln_variant_openmp<Variant::Fish>      (p, out, on_row_done); break;
        case Variant::Vase:       render_ln_variant_openmp<Variant::Vase>      (p, out, on_row_done); break;
        case Variant::Bird:       render_ln_variant_openmp<Variant::Bird>      (p, out, on_row_done); break;
        case Variant::Mask:       render_ln_variant_openmp<Variant::Mask>      (p, out, on_row_done); break;
        case Variant::Ship:       render_ln_variant_openmp<Variant::Ship>      (p, out, on_row_done); break;
        case Variant::SinZ:       render_ln_variant_openmp<Variant::SinZ>      (p, out, on_row_done); break;
        case Variant::CosZ:       render_ln_variant_openmp<Variant::CosZ>      (p, out, on_row_done); break;
        case Variant::ExpZ:       render_ln_variant_openmp<Variant::ExpZ>      (p, out, on_row_done); break;
        case Variant::SinhZ:      render_ln_variant_openmp<Variant::SinhZ>     (p, out, on_row_done); break;
        case Variant::CoshZ:      render_ln_variant_openmp<Variant::CoshZ>     (p, out, on_row_done); break;
        case Variant::TanZ:       render_ln_variant_openmp<Variant::TanZ>      (p, out, on_row_done); break;
        case Variant::Custom:     render_ln_variant_openmp<Variant::Mandelbrot>(p, out, on_row_done); break;
    }
}

bool should_try_cuda(const LnMapParams& p) {
#if USE_CUDA_LN_MAP
    if (!ln_map_variant_supported_by_simd(p.variant)) return false;
    if (p.engine == "cuda") return fsd_cuda::cuda_ln_map_available();
    if (p.engine != "auto") return false;
    const auto caps = runtime_capabilities();
    return caps.cuda_runtime && !caps.cuda_low_end;
#else
    (void)p;
    return false;
#endif
}

bool should_try_avx512(const LnMapParams& p) {
    if (!ln_map_variant_supported_by_simd(p.variant)) return false;
    if (p.engine == "avx512") return avx512_available();
    if (p.engine != "auto") return false;
    return avx512_available();
}

} // namespace

bool ln_map_variant_supported_by_simd(Variant v) {
    const int id = static_cast<int>(v);
    return id >= 0 && id <= 9;
}

LnMapStats render_ln_map_openmp(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    if (out.empty() || out.rows != p.height_t || out.cols != p.width_s || out.type() != CV_8UC3) {
        out.create(p.height_t, p.width_s, CV_8UC3);
    }
    const auto t0 = std::chrono::steady_clock::now();
    dispatch_openmp(p, out, on_row_done);
    const auto t1 = std::chrono::steady_clock::now();
    LnMapStats stats;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.pixel_count = p.width_s * p.height_t;
    stats.engine_used = "openmp";
    stats.scalar_used = "fp64";
    return stats;
}

LnMapStats render_ln_map(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    if (should_try_cuda(p)) {
#if USE_CUDA_LN_MAP
        try {
            fsd_cuda::CudaLnMapParams cp;
            cp.julia = p.julia;
            cp.center_re = p.center_re;
            cp.center_im = p.center_im;
            cp.julia_re = p.julia_re;
            cp.julia_im = p.julia_im;
            cp.width_s = p.width_s;
            cp.height_t = p.height_t;
            cp.iterations = p.iterations;
            cp.bailout = p.bailout;
            cp.bailout_sq = p.bailout_sq;
            cp.variant_id = static_cast<int>(p.variant);
            cp.colormap_id = static_cast<int>(p.colormap);
            const auto stats = fsd_cuda::cuda_render_ln_map(cp, out);
            if (on_row_done) on_row_done(p.height_t);
            return {stats.elapsed_ms, p.width_s * p.height_t, "cuda", "fp64"};
        } catch (...) {
            if (p.engine == "cuda") throw;
        }
#endif
    }

    if (should_try_avx512(p)) {
        try {
            return render_ln_map_avx512(p, out, on_row_done);
        } catch (...) {
            if (p.engine == "avx512") throw;
        }
    }

    return render_ln_map_openmp(p, out, on_row_done);
}

} // namespace fsd::compute
