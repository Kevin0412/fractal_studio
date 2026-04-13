// compute/map_kernel.cpp
//
// OpenMP-parallel map renderer. For each pixel, iterates the chosen variant
// under the chosen metric and writes a BGR byte triple into the output Mat.
//
// Supports two scalar types:
//   fp64  — std::double (default, good to ~1e-13 zoom depth)
//   fx64  — Fx64 fixed-point 1s·6i·57f (good to ~1e-17, ~4 extra magnitudes)
//
// The variant is dispatched at compile time via `variant_step<V,S>`.

#include "map_kernel.hpp"
#include "map_kernel_avx512.hpp"
#include "escape_time.hpp"
#include "complex.hpp"
#include "scalar/fx64.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/map_kernel.cuh"
#  define USE_CUDA 1
#else
#  define USE_CUDA 0
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <chrono>
#include <cmath>
#include <vector>

namespace fsd::compute {

namespace {

// Normalize a metric sample into [0,1] for colorize_field_bgr.
// For escape metric this function is unused (colorize_escape_bgr takes
// iter+max_iter directly).
inline double normalize_field(const IterResult& r, Metric m, double bailout) {
    switch (m) {
        case Metric::MinAbs:
            // min|z_n| is in [0, bailout]; map to [0,1].
            if (!std::isfinite(r.min_abs)) return 1.0;
            return std::min(1.0, r.min_abs / bailout);
        case Metric::MaxAbs:
            if (r.max_abs <= 0.0) return 0.0;
            return std::min(1.0, r.max_abs / bailout);
        case Metric::Envelope:
            // Combine min+max into a single brightness for now; the dedicated
            // envelope endpoint in P2 emits two fields.
            if (!std::isfinite(r.min_abs)) return 1.0;
            return std::min(1.0, 0.5 * (r.min_abs + r.max_abs) / bailout);
        case Metric::MinPairwiseDist:
            if (!std::isfinite(r.extra)) return 1.0;
            return std::min(1.0, r.extra / bailout);
        default:
            return 0.0;
    }
}

// Generic OpenMP kernel templated on both Variant and Scalar.
template <Variant V, typename S>
void render_variant_impl(const MapParams& p, cv::Mat& out) {
    const int W = p.width;
    const int H = p.height;
    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout * p.bailout;

    const S jre = scalar_from_double<S>(p.julia_re);
    const S jim = scalar_from_double<S>(p.julia_im);
    const Cx<S> c_const{jre, jim};

    #pragma omp parallel
    {
        std::vector<Cx<S>> orbit;
        orbit.reserve(static_cast<size_t>(p.pairwise_cap));

        #pragma omp for schedule(dynamic, 4)
        for (int y = 0; y < H; y++) {
            uint8_t* row = out.ptr<uint8_t>(y);
            const double im_d = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
            const S im = scalar_from_double<S>(im_d);
            for (int x = 0; x < W; x++) {
                const double re_d = re_min + (static_cast<double>(x) + 0.5) / W * span_re;
                const S re = scalar_from_double<S>(re_d);

                Cx<S> z0;
                Cx<S> c;
                if (p.julia) {
                    z0 = Cx<S>{re, im};
                    c  = c_const;
                } else {
                    z0 = Cx<S>{scalar_from_double<S>(0.0), scalar_from_double<S>(0.0)};
                    c  = Cx<S>{re, im};
                }

                const IterResult r = iterate<V, S>(
                    z0, c, p.iterations, bail2, p.metric, p.pairwise_cap, orbit
                );

                uint8_t* px = row + 3 * x;
                if (p.metric == Metric::Escape) {
                    const int    iter = r.escaped ? r.iter : p.iterations;
                    const double norm = r.escaped ? r.norm : 0.0;
                    colorize_escape_bgr(iter, p.iterations, p.colormap, norm, p.smooth, px[0], px[1], px[2]);
                } else if (p.colormap == Colormap::HsRainbow) {
                    // HsRainbow always uses the log-scale formula directly.
                    double fv = 0.0;
                    switch (p.metric) {
                        case Metric::MinAbs:
                            fv = std::isfinite(r.min_abs) ? r.min_abs : 0.0; break;
                        case Metric::MaxAbs:
                            fv = (r.max_abs > 0.0) ? r.max_abs : 0.0; break;
                        case Metric::Envelope:
                            fv = std::isfinite(r.min_abs)
                                ? 0.5 * (r.min_abs + r.max_abs) : 0.0; break;
                        case Metric::MinPairwiseDist:
                            fv = std::isfinite(r.extra) ? r.extra : 0.0; break;
                        default: fv = 0.0; break;
                    }
                    colorize_field_hs_bgr(fv, px[0], px[1], px[2]);
                } else if (p.smooth) {
                    // Raw field value for ln-smooth coloring.
                    double fv = 0.0;
                    switch (p.metric) {
                        case Metric::MinAbs:
                            fv = std::isfinite(r.min_abs) ? r.min_abs : 0.0; break;
                        case Metric::MaxAbs:
                            fv = (r.max_abs > 0.0) ? r.max_abs : 0.0; break;
                        case Metric::Envelope:
                            fv = std::isfinite(r.min_abs)
                                ? 0.5 * (r.min_abs + r.max_abs) : 0.0; break;
                        case Metric::MinPairwiseDist:
                            fv = std::isfinite(r.extra) ? r.extra : 0.0; break;
                        default: fv = 0.0; break;
                    }
                    colorize_field_smooth_bgr(fv, p.colormap, px[0], px[1], px[2]);
                } else {
                    const double v01 = normalize_field(r, p.metric, p.bailout);
                    colorize_field_bgr(v01, p.colormap, px[0], px[1], px[2]);
                }
            }
        }
    }
}

// Variant dispatch helpers — one for fp64, one for fx64.
template <Variant V>
void render_variant(const MapParams& p, cv::Mat& out) {
    render_variant_impl<V, double>(p, out);
}

template <Variant V>
void render_variant_fx64(const MapParams& p, cv::Mat& out) {
    render_variant_impl<V, Fx64>(p, out);
}

} // namespace

// Determine whether to use Fx64 based on params.
static bool use_fx64(const MapParams& p) {
    if (p.scalar_type == "fx64") return true;
    if (p.scalar_type == "fp64") return false;
    // "auto": switch to Fx64 when scale < 1e-13 (fp64 loses too much precision).
    return p.scale < 1e-13;
}

// Dispatch fp64 variants
static void dispatch_fp64(const MapParams& p, cv::Mat& out) {
    switch (p.variant) {
        case Variant::Mandelbrot: render_variant<Variant::Mandelbrot>(p, out); break;
        case Variant::Tri:        render_variant<Variant::Tri>(p, out);        break;
        case Variant::Boat:       render_variant<Variant::Boat>(p, out);       break;
        case Variant::Duck:       render_variant<Variant::Duck>(p, out);       break;
        case Variant::Bell:       render_variant<Variant::Bell>(p, out);       break;
        case Variant::Fish:       render_variant<Variant::Fish>(p, out);       break;
        case Variant::Vase:       render_variant<Variant::Vase>(p, out);       break;
        case Variant::Bird:       render_variant<Variant::Bird>(p, out);       break;
        case Variant::Mask:       render_variant<Variant::Mask>(p, out);       break;
        case Variant::Ship:       render_variant<Variant::Ship>(p, out);       break;
    }
}

// Dispatch fx64 variants
static void dispatch_fx64(const MapParams& p, cv::Mat& out) {
    switch (p.variant) {
        case Variant::Mandelbrot: render_variant_fx64<Variant::Mandelbrot>(p, out); break;
        case Variant::Tri:        render_variant_fx64<Variant::Tri>(p, out);        break;
        case Variant::Boat:       render_variant_fx64<Variant::Boat>(p, out);       break;
        case Variant::Duck:       render_variant_fx64<Variant::Duck>(p, out);       break;
        case Variant::Bell:       render_variant_fx64<Variant::Bell>(p, out);       break;
        case Variant::Fish:       render_variant_fx64<Variant::Fish>(p, out);       break;
        case Variant::Vase:       render_variant_fx64<Variant::Vase>(p, out);       break;
        case Variant::Bird:       render_variant_fx64<Variant::Bird>(p, out);       break;
        case Variant::Mask:       render_variant_fx64<Variant::Mask>(p, out);       break;
        case Variant::Ship:       render_variant_fx64<Variant::Ship>(p, out);       break;
    }
}

MapStats render_map(const MapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    const bool fx  = use_fx64(p);

    // smooth coloring needs per-pixel |z|² which the AVX-512/CUDA paths don't track.
    // Fall through to OpenMP which has access to IterResult.norm.
    const bool needs_norm = p.smooth;

    // CUDA path: Mandelbrot + Escape only (no Julia, no smooth).
#if USE_CUDA
    const bool can_cuda = !needs_norm
                       && !p.julia
                       && (p.engine == "cuda" || p.engine == "auto" || p.engine == "hybrid")
                       && (p.variant == Variant::Mandelbrot)
                       && (p.metric == Metric::Escape)
                       && fsd_cuda::cuda_available();
    if (can_cuda) {
        fsd_cuda::CudaMapParams cp;
        cp.center_re  = p.center_re;
        cp.center_im  = p.center_im;
        cp.scale      = p.scale;
        cp.width      = p.width;
        cp.height     = p.height;
        cp.iterations = p.iterations;
        cp.bailout    = p.bailout;
        cp.scalar_type  = fx ? "fx64" : "fp64";
        cp.colormap_id  = static_cast<int>(p.colormap);
        auto cs = fsd_cuda::cuda_render_map(cp, out);
        MapStats s;
        s.elapsed_ms  = cs.elapsed_ms;
        s.pixel_count = p.width * p.height;
        s.scalar_used = cs.scalar_used;
        s.engine_used = "cuda";
        return s;
    }
#endif

    // AVX-512 path: only Mandelbrot + Escape metric currently (most common case).
    // Other variants, non-escape metrics, and smooth mode fall through to OpenMP.
    const bool can_avx = !needs_norm
                      && (p.engine == "avx512" || p.engine == "auto" || p.engine == "hybrid")
                      && (p.variant == Variant::Mandelbrot)
                      && (p.metric == Metric::Escape)
                      && avx512_available();

    if (can_avx) {
        auto s = fx ? render_map_avx512_fx64(p, out)
                    : render_map_avx512_fp64(p, out);
        s.pixel_count = p.width * p.height;
        s.scalar_used = fx ? "fx64" : "fp64";
        s.engine_used = "avx512";
        return s;
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (fx) {
        dispatch_fx64(p, out);
    } else {
        dispatch_fp64(p, out);
    }

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count  = p.width * p.height;
    s.scalar_used  = fx ? "fx64" : "fp64";
    s.engine_used  = "openmp";
    return s;
}

} // namespace fsd::compute
