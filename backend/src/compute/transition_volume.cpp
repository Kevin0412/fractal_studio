// compute/transition_volume.cpp

#include "transition_volume.hpp"
#include "engine_select.hpp"
#include "parallel.hpp"
#include "transition_volume_avx2.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/transition_volume.cuh"
#  define USE_CUDA_TRANSITION 1
#else
#  define USE_CUDA_TRANSITION 0
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fsd::compute {

namespace {

inline float transition_real_projection_f32(Variant v, float x2, float axis2) {
    const bool post_abs =
        v == Variant::Fish || v == Variant::Vase || v == Variant::Bird ||
        v == Variant::Mask || v == Variant::Ship;
    float q = x2 - axis2;
    if (post_abs) q = std::fabs(q);
    return q;
}

inline float transition_imag_projection_f32(Variant v, float x, float axis) {
    const bool abs_x =
        v == Variant::Boat || v == Variant::Bell || v == Variant::Ship;
    const bool abs_axis =
        v == Variant::Boat || v == Variant::Duck || v == Variant::Mask || v == Variant::Bird;
    const bool neg =
        v == Variant::Tri || v == Variant::Bell || v == Variant::Vase || v == Variant::Ship;
    float a = abs_x ? std::fabs(x) : x;
    float b = abs_axis ? std::fabs(axis) : axis;
    float q = 2.0f * a * b;
    return neg ? -q : q;
}

std::string select_transition_engine(const TransitionVolumeParams& p) {
    const RuntimeCapabilities caps = runtime_capabilities();
    const long long voxels = static_cast<long long>(p.resolution) * p.resolution * p.resolution;
    const long long work = voxels * static_cast<long long>(std::max(1, p.iterations));
    const bool large = work >= 250000000LL;

    if (p.engine == "cuda") return caps.cuda_runtime ? "cuda" : "openmp";
    if (p.engine == "avx2") return (caps.avx2_compiled && caps.avx2_runtime && caps.fma_runtime) ? "avx2" : "openmp";
    if (p.engine == "avx512") return "openmp";
    if (p.engine == "hybrid") return (large && caps.cuda_runtime) ? "hybrid" : "openmp";
    if (p.engine == "openmp") return "openmp";

    if (large && caps.cuda_runtime && !caps.cuda_low_end) return "hybrid";
    if (caps.avx2_compiled && caps.avx2_runtime && caps.fma_runtime) return "avx2";
    return "openmp";
}

} // namespace

McField buildTransitionVolume(const TransitionVolumeParams& p) {
    if (!variant_supports_axis_transition(p.from_variant) ||
        !variant_supports_axis_transition(p.to_variant)) {
        throw std::runtime_error("transition variants must be quadratic Mandelbrot-family variants");
    }

    const int N = std::max(4, std::min(1024, p.resolution));
    McField field;
    field.Nx = field.Ny = field.Nz = N;
    field.data.assign(static_cast<size_t>(N) * N * N, 1.0f);
    field.scalar_used = "fp32";
    const std::string selected_engine = select_transition_engine(p);
    field.engine_used = selected_engine == "openmp" ? "openmp_fp32" : selected_engine + "_fp32_openmp_fallback";

#if USE_CUDA_TRANSITION
    if (selected_engine == "cuda" || selected_engine == "hybrid") {
        try {
            fsd_cuda::CudaTransitionVolumeParams cp;
            cp.center_x = static_cast<float>(p.centerX);
            cp.center_y = static_cast<float>(p.centerY);
            cp.center_z = static_cast<float>(p.centerZ);
            cp.extent = static_cast<float>(p.extent);
            cp.resolution = N;
            cp.iterations = p.iterations;
            cp.bailout = static_cast<float>(p.bailout);
            cp.bailout_sq = static_cast<float>(p.bailout_sq);
            cp.from_variant = static_cast<int>(p.from_variant);
            cp.to_variant = static_cast<int>(p.to_variant);
            fsd_cuda::cuda_build_transition_volume(cp, field.data);
            field.engine_used = selected_engine == "hybrid" ? "hybrid_cuda_fp32" : "cuda_fp32";
            return field;
        } catch (...) {
            field.data.assign(static_cast<size_t>(N) * N * N, 1.0f);
            field.engine_used = selected_engine + "_fp32_openmp_fallback";
        }
    }
#endif

    if (selected_engine == "avx2") {
        if (buildTransitionVolumeAvx2(p, field)) {
            return field;
        }
        field.data.assign(static_cast<size_t>(N) * N * N, 1.0f);
        field.engine_used = "avx2_fp32_openmp_fallback";
    }

    const float span = static_cast<float>(p.extent * 2.0);
    const float xmin = static_cast<float>(p.centerX - p.extent);
    const float ymin = static_cast<float>(p.centerY - p.extent);
    const float zmin = static_cast<float>(p.centerZ - p.extent);
    const float bail2 = static_cast<float>(p.bailout_sq);
    const float bailout = static_cast<float>(p.bailout);

    const int maxIter = p.iterations;
    const int thread_count = default_render_threads();

    #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 1)
    for (int zi = 0; zi < N; zi++) {
        const float z0 = zmin + (static_cast<float>(zi) + 0.5f) / static_cast<float>(N) * span;
        for (int yi = 0; yi < N; yi++) {
            const float y0 = ymin + (static_cast<float>(yi) + 0.5f) / static_cast<float>(N) * span;
            for (int xi = 0; xi < N; xi++) {
                const float x0 = xmin + (static_cast<float>(xi) + 0.5f) / static_cast<float>(N) * span;

                float x = x0, y = y0, z = z0;
                int iter = 0;
                bool escaped = false;
                for (; iter < maxIter; iter++) {
                    const float x2 = x * x;
                    const float nx =
                        transition_real_projection_f32(p.from_variant, x2, y * y)
                      + transition_real_projection_f32(p.to_variant,   x2, z * z)
                      - x2 + x0;
                    const float ny = transition_imag_projection_f32(p.from_variant, x, y) + y0;
                    const float nz = transition_imag_projection_f32(p.to_variant,   x, z) + z0;
                    x = nx; y = ny; z = nz;
                    const bool finite_xyz = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
                    const float n2 = finite_xyz
                        ? (x*x + y*y + z*z)
                        : std::numeric_limits<float>::infinity();
                    if (!finite_xyz || n2 > bail2) {
                        escaped = true;
                        break;
                    }
                }

                // Scalar field: iso = 0.5, inside < 0.5, outside >= 0.5.
                //
                // Outside: v in [0.5, 1.0] — faster escape → closer to 1.
                // Inside:  v in [0.0, 0.48] — final orbit magnitude gives a depth
                //          gradient so the voxel renderer can shade surface vs.
                //          interior voxels differently (boundary = bright, deep = dark).
                float v = 0.0f;
                if (escaped) {
                    v = 0.5f + 0.5f * (static_cast<float>(iter) / static_cast<float>(maxIter));
                } else {
                    const bool finite_xyz = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
                    const float finalMag = finite_xyz ? std::sqrt(x*x + y*y + z*z) : bailout;
                    // Normalize to [0, 0.48) — near-boundary orbits reach higher mag.
                    v = (finalMag / bailout) * 0.48f;
                }

                const size_t idx = static_cast<size_t>(xi) +
                                   static_cast<size_t>(N) *
                                   (static_cast<size_t>(yi) + static_cast<size_t>(N) * static_cast<size_t>(zi));
                field.data[idx] = v;
            }
        }
    }

    return field;
}

} // namespace fsd::compute
