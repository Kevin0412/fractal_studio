// compute/transition_volume_avx2.cpp

#include "transition_volume_avx2.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace fsd::compute {

#if defined(__AVX2__) && defined(__FMA__)

namespace {

struct FoldRules {
    bool post_abs_real = false;
    bool abs_x = false;
    bool abs_axis = false;
    bool neg_imag = false;
};

FoldRules fold_rules(Variant v) {
    FoldRules r;
    r.post_abs_real =
        v == Variant::Fish || v == Variant::Vase || v == Variant::Bird ||
        v == Variant::Mask || v == Variant::Ship;
    r.abs_x = v == Variant::Boat || v == Variant::Bell || v == Variant::Ship;
    r.abs_axis = v == Variant::Boat || v == Variant::Duck ||
                 v == Variant::Mask || v == Variant::Bird;
    r.neg_imag = v == Variant::Tri || v == Variant::Bell ||
                 v == Variant::Vase || v == Variant::Ship;
    return r;
}

inline __m256 avx2_abs_ps(__m256 v) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign, v);
}

inline __m256 avx2_lane_mask_ps(int mask) {
    const __m256i bits = _mm256_set_epi32(
        (mask & 128) ? -1 : 0,
        (mask & 64) ? -1 : 0,
        (mask & 32) ? -1 : 0,
        (mask & 16) ? -1 : 0,
        (mask & 8) ? -1 : 0,
        (mask & 4) ? -1 : 0,
        (mask & 2) ? -1 : 0,
        (mask & 1) ? -1 : 0);
    return _mm256_castsi256_ps(bits);
}

inline __m256 real_projection(__m256 x2, __m256 axis2, const FoldRules& r) {
    __m256 q = _mm256_sub_ps(x2, axis2);
    return r.post_abs_real ? avx2_abs_ps(q) : q;
}

inline __m256 imag_projection(__m256 x, __m256 axis, const FoldRules& r) {
    const __m256 a = r.abs_x ? avx2_abs_ps(x) : x;
    const __m256 b = r.abs_axis ? avx2_abs_ps(axis) : axis;
    __m256 q = _mm256_add_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(a, b));
    return r.neg_imag ? _mm256_sub_ps(_mm256_setzero_ps(), q) : q;
}

} // namespace

bool buildTransitionVolumeAvx2(const TransitionVolumeParams& p, McField& field) {
    if (!avx2_available() || !fma_available()) return false;
    if (!variant_supports_axis_transition(p.from_variant) ||
        !variant_supports_axis_transition(p.to_variant)) {
        return false;
    }

    const int N = std::max(4, std::min(1024, p.resolution));
    field.Nx = field.Ny = field.Nz = N;
    field.data.assign(static_cast<size_t>(N) * N * N, 1.0f);
    field.scalar_used = "fp32";
    field.engine_used = "avx2_fp32";

    const FoldRules from = fold_rules(p.from_variant);
    const FoldRules to = fold_rules(p.to_variant);
    const float span = static_cast<float>(p.extent * 2.0);
    const float xmin = static_cast<float>(p.centerX - p.extent);
    const float ymin = static_cast<float>(p.centerY - p.extent);
    const float zmin = static_cast<float>(p.centerZ - p.extent);
    const float bail2 = static_cast<float>(p.bailout_sq);
    const float bailout = static_cast<float>(p.bailout);
    const int maxIter = p.iterations;
    const int thread_count = default_render_threads();

    const __m256 lane_offsets = _mm256_set_ps(7.5f, 6.5f, 5.5f, 4.5f, 3.5f, 2.5f, 1.5f, 0.5f);
    const __m256 vN = _mm256_set1_ps(static_cast<float>(N));
    const __m256 vspan = _mm256_set1_ps(span);
    const __m256 vxmin = _mm256_set1_ps(xmin);
    const __m256 vbail2 = _mm256_set1_ps(bail2);
    #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1)
    for (int zi = 0; zi < N; ++zi) {
        const float z0s = zmin + (static_cast<float>(zi) + 0.5f) / static_cast<float>(N) * span;
        const __m256 vz0 = _mm256_set1_ps(z0s);
        for (int yi = 0; yi < N; ++yi) {
            const float y0s = ymin + (static_cast<float>(yi) + 0.5f) / static_cast<float>(N) * span;
            const __m256 vy0 = _mm256_set1_ps(y0s);
            for (int xi = 0; xi < N; xi += 8) {
                int lane_mask = 0;
                for (int k = 0; k < 8; ++k) {
                    if (xi + k < N) lane_mask |= (1 << k);
                }
                const __m256 vxi = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(xi)), lane_offsets);
                const __m256 vx0 = _mm256_fmadd_ps(_mm256_div_ps(vxi, vN), vspan, vxmin);

                __m256 vx = vx0;
                __m256 vy = vy0;
                __m256 vz = vz0;
                int active = lane_mask;
                int iters[8] = {
                    maxIter, maxIter, maxIter, maxIter,
                    maxIter, maxIter, maxIter, maxIter
                };

                for (int iter = 0; iter < maxIter && active; ++iter) {
                    const __m256 active_vec = avx2_lane_mask_ps(active);
                    const __m256 x2 = _mm256_mul_ps(vx, vx);
                    const __m256 y2 = _mm256_mul_ps(vy, vy);
                    const __m256 z2 = _mm256_mul_ps(vz, vz);
                    const __m256 nx = _mm256_add_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(real_projection(x2, y2, from),
                                          real_projection(x2, z2, to)),
                            x2),
                        vx0);
                    const __m256 ny = _mm256_add_ps(imag_projection(vx, vy, from), vy0);
                    const __m256 nz = _mm256_add_ps(imag_projection(vx, vz, to), vz0);

                    vx = _mm256_blendv_ps(vx, nx, active_vec);
                    vy = _mm256_blendv_ps(vy, ny, active_vec);
                    vz = _mm256_blendv_ps(vz, nz, active_vec);

                    const __m256 n2 = _mm256_fmadd_ps(vz, vz,
                        _mm256_fmadd_ps(vy, vy, _mm256_mul_ps(vx, vx)));
                    const __m256 escaped_radius = _mm256_cmp_ps(n2, vbail2, _CMP_GT_OQ);
                    const __m256 escaped_nan = _mm256_cmp_ps(n2, n2, _CMP_UNORD_Q);
                    const int escaped = _mm256_movemask_ps(_mm256_or_ps(escaped_radius, escaped_nan)) & active;
                    if (escaped) {
                        for (int k = 0; k < 8; ++k) {
                            if (escaped & (1 << k)) iters[k] = iter;
                        }
                        active &= ~escaped;
                    }
                }

                alignas(32) float xs[8], ys[8], zs[8];
                _mm256_store_ps(xs, vx);
                _mm256_store_ps(ys, vy);
                _mm256_store_ps(zs, vz);
                for (int k = 0; k < 8 && xi + k < N; ++k) {
                    float value = 0.0f;
                    if (iters[k] < maxIter) {
                        value = 0.5f + 0.5f * (static_cast<float>(iters[k]) / static_cast<float>(maxIter));
                    } else {
                        const float mag2 = xs[k] * xs[k] + ys[k] * ys[k] + zs[k] * zs[k];
                        const float finalMag = std::isfinite(mag2) ? std::sqrt(mag2) : bailout;
                        value = std::min(0.48f, (finalMag / bailout) * 0.48f);
                    }
                    const size_t idx = static_cast<size_t>(xi + k) +
                        static_cast<size_t>(N) *
                        (static_cast<size_t>(yi) + static_cast<size_t>(N) * static_cast<size_t>(zi));
                    field.data[idx] = value;
                }
            }
        }
    }

    return true;
}

#else

bool buildTransitionVolumeAvx2(const TransitionVolumeParams& p, McField& field) {
    (void)p; (void)field;
    return false;
}

#endif

} // namespace fsd::compute
