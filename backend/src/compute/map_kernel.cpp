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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
        case Variant::SinZ:       render_variant<Variant::SinZ>(p, out);       break;
        case Variant::CosZ:       render_variant<Variant::CosZ>(p, out);       break;
        case Variant::ExpZ:       render_variant<Variant::ExpZ>(p, out);       break;
        case Variant::SinhZ:      render_variant<Variant::SinhZ>(p, out);      break;
        case Variant::CoshZ:      render_variant<Variant::CoshZ>(p, out);      break;
        case Variant::TanZ:       render_variant<Variant::TanZ>(p, out);       break;
        default: break;  // Variant::Custom is intercepted before this dispatch
    }
}

// Dispatch fx64 variants (trig variants fall back to fp64 via apply_trig)
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
        // Trig variants: apply_trig already casts Fx64 → double internally.
        case Variant::SinZ:       render_variant_fx64<Variant::SinZ>(p, out);       break;
        case Variant::CosZ:       render_variant_fx64<Variant::CosZ>(p, out);       break;
        case Variant::ExpZ:       render_variant_fx64<Variant::ExpZ>(p, out);       break;
        case Variant::SinhZ:      render_variant_fx64<Variant::SinhZ>(p, out);      break;
        case Variant::CoshZ:      render_variant_fx64<Variant::CoshZ>(p, out);      break;
        case Variant::TanZ:       render_variant_fx64<Variant::TanZ>(p, out);       break;
        default: break;  // Variant::Custom is intercepted before this dispatch
    }
}

// ─── Custom variant renderer (OpenMP, function pointer) ──────────────────────
//
// Only used when p.variant == Variant::Custom && p.custom_step_fn != nullptr.
// Falls back to escape metric colorization (min_abs/max_abs/envelope/pairwise
// are not tracked since the custom step fn has no access to orbit buffers).

static MapStats render_custom_openmp(const MapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    CustomStepFn fn = p.custom_step_fn;

    const int W = p.width, H = p.height;
    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout * p.bailout;
    const double jre     = p.julia_re;
    const double jim     = p.julia_im;

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < H; y++) {
        uint8_t* row = out.ptr<uint8_t>(y);
        const double im_c = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
        for (int x = 0; x < W; x++) {
            const double re_c = re_min + (static_cast<double>(x) + 0.5) / W * span_re;

            double zr, zi, cr, ci;
            if (p.julia) { zr = re_c; zi = im_c; cr = jre; ci = jim; }
            else          { zr = 0.0; zi = 0.0;  cr = re_c; ci = im_c; }

            int   it    = 0;
            double norm2 = 0.0;
            for (; it < p.iterations; it++) {
                double nr = 0.0, ni = 0.0;
                fn(zr, zi, cr, ci, &nr, &ni);
                zr = nr; zi = ni;
                norm2 = zr * zr + zi * zi;
                if (norm2 > bail2) break;
            }

            const bool escaped = (norm2 > bail2);
            uint8_t* px = row + 3 * x;
            colorize_escape_bgr(
                escaped ? it : p.iterations,
                p.iterations,
                p.colormap,
                escaped ? norm2 : 0.0,
                p.smooth,
                px[0], px[1], px[2]
            );
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = W * H;
    s.scalar_used = "fp64";
    s.engine_used = "openmp";
    return s;
}

// Field variant for custom formula — fills FieldOutput with escape metric data.
static MapStats render_custom_field_openmp(const MapParams& p, FieldOutput& fo) {
    fo.width  = p.width;
    fo.height = p.height;
    fo.metric = Metric::Escape;  // custom always uses escape metric for field

    CustomStepFn fn = p.custom_step_fn;

    const int W = p.width, H = p.height;
    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout * p.bailout;
    const double jre     = p.julia_re;
    const double jim     = p.julia_im;

    fo.iter_u32.assign(static_cast<size_t>(W) * H, 0u);
    fo.norm_f32.assign(static_cast<size_t>(W) * H, 0.0f);

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < H; y++) {
        const double im_c = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
        for (int x = 0; x < W; x++) {
            const double re_c = re_min + (static_cast<double>(x) + 0.5) / W * span_re;

            double zr, zi, cr, ci;
            if (p.julia) { zr = re_c; zi = im_c; cr = jre; ci = jim; }
            else          { zr = 0.0; zi = 0.0;  cr = re_c; ci = im_c; }

            int   it    = 0;
            double norm2 = 0.0;
            for (; it < p.iterations; it++) {
                double nr = 0.0, ni = 0.0;
                fn(zr, zi, cr, ci, &nr, &ni);
                zr = nr; zi = ni;
                norm2 = zr * zr + zi * zi;
                if (norm2 > bail2) break;
            }

            const bool escaped = (norm2 > bail2);
            const size_t idx = static_cast<size_t>(y) * W + x;
            fo.iter_u32[idx] = escaped ? static_cast<uint32_t>(it) : static_cast<uint32_t>(p.iterations);
            fo.norm_f32[idx] = escaped ? static_cast<float>(norm2) : 0.0f;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = W * H;
    s.scalar_used = "fp64";
    s.engine_used = "openmp";
    fo.scalar_used  = "fp64";
    fo.engine_used  = "openmp";
    return s;
}

// ─── Raw field renderer (always OpenMP, no colorization) ─────────────────────

namespace {

// Extract the raw metric value from an IterResult for non-escape metrics.
inline double extract_raw_field(const IterResult& r, Metric m) {
    switch (m) {
        case Metric::MinAbs:
            return std::isfinite(r.min_abs) ? r.min_abs : 0.0;
        case Metric::MaxAbs:
            return r.max_abs;
        case Metric::Envelope:
            return std::isfinite(r.min_abs) ? 0.5 * (r.min_abs + r.max_abs) : 0.0;
        case Metric::MinPairwiseDist:
            return std::isfinite(r.extra) ? r.extra : 0.0;
        default:
            return 0.0;
    }
}

template <Variant V, typename S>
void field_variant_impl(const MapParams& p, FieldOutput& out) {
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

    const bool is_escape = (p.metric == Metric::Escape);
    if (is_escape) {
        out.iter_u32.assign(static_cast<size_t>(W) * H, 0u);
        out.norm_f32.assign(static_cast<size_t>(W) * H, 0.0f);
    } else {
        out.field_f64.assign(static_cast<size_t>(W) * H, 0.0);
    }

    #pragma omp parallel
    {
        std::vector<Cx<S>> orbit;
        orbit.reserve(static_cast<size_t>(p.pairwise_cap));

        #pragma omp for schedule(dynamic, 4)
        for (int y = 0; y < H; y++) {
            const double im_d = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
            const S im = scalar_from_double<S>(im_d);
            for (int x = 0; x < W; x++) {
                const double re_d = re_min + (static_cast<double>(x) + 0.5) / W * span_re;
                const S re = scalar_from_double<S>(re_d);

                Cx<S> z0, c;
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

                const size_t idx = static_cast<size_t>(y) * W + x;
                if (is_escape) {
                    out.iter_u32[idx] = r.escaped
                        ? static_cast<uint32_t>(r.iter)
                        : static_cast<uint32_t>(p.iterations);
                    out.norm_f32[idx] = r.escaped ? static_cast<float>(r.norm) : 0.0f;
                } else {
                    out.field_f64[idx] = extract_raw_field(r, p.metric);
                }
            }
        }
    }

    if (!is_escape) {
        double lo =  std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (double v : out.field_f64) {
            if (std::isfinite(v)) {
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
        }
        out.field_min = std::isfinite(lo) ? lo : 0.0;
        out.field_max = std::isfinite(hi) ? hi : 1.0;
    }
}

template <Variant V>
void field_variant_fp64(const MapParams& p, FieldOutput& out) {
    field_variant_impl<V, double>(p, out);
}

template <Variant V>
void field_variant_fx64(const MapParams& p, FieldOutput& out) {
    field_variant_impl<V, Fx64>(p, out);
}

void dispatch_field_fp64(const MapParams& p, FieldOutput& out) {
    switch (p.variant) {
        case Variant::Mandelbrot: field_variant_fp64<Variant::Mandelbrot>(p, out); break;
        case Variant::Tri:        field_variant_fp64<Variant::Tri>       (p, out); break;
        case Variant::Boat:       field_variant_fp64<Variant::Boat>      (p, out); break;
        case Variant::Duck:       field_variant_fp64<Variant::Duck>      (p, out); break;
        case Variant::Bell:       field_variant_fp64<Variant::Bell>      (p, out); break;
        case Variant::Fish:       field_variant_fp64<Variant::Fish>      (p, out); break;
        case Variant::Vase:       field_variant_fp64<Variant::Vase>      (p, out); break;
        case Variant::Bird:       field_variant_fp64<Variant::Bird>      (p, out); break;
        case Variant::Mask:       field_variant_fp64<Variant::Mask>      (p, out); break;
        case Variant::Ship:       field_variant_fp64<Variant::Ship>      (p, out); break;
        case Variant::SinZ:       field_variant_fp64<Variant::SinZ>      (p, out); break;
        case Variant::CosZ:       field_variant_fp64<Variant::CosZ>      (p, out); break;
        case Variant::ExpZ:       field_variant_fp64<Variant::ExpZ>      (p, out); break;
        case Variant::SinhZ:      field_variant_fp64<Variant::SinhZ>     (p, out); break;
        case Variant::CoshZ:      field_variant_fp64<Variant::CoshZ>     (p, out); break;
        case Variant::TanZ:       field_variant_fp64<Variant::TanZ>      (p, out); break;
        default: break;  // Variant::Custom intercepted before this dispatch
    }
}

void dispatch_field_fx64(const MapParams& p, FieldOutput& out) {
    switch (p.variant) {
        case Variant::Mandelbrot: field_variant_fx64<Variant::Mandelbrot>(p, out); break;
        case Variant::Tri:        field_variant_fx64<Variant::Tri>       (p, out); break;
        case Variant::Boat:       field_variant_fx64<Variant::Boat>      (p, out); break;
        case Variant::Duck:       field_variant_fx64<Variant::Duck>      (p, out); break;
        case Variant::Bell:       field_variant_fx64<Variant::Bell>      (p, out); break;
        case Variant::Fish:       field_variant_fx64<Variant::Fish>      (p, out); break;
        case Variant::Vase:       field_variant_fx64<Variant::Vase>      (p, out); break;
        case Variant::Bird:       field_variant_fx64<Variant::Bird>      (p, out); break;
        case Variant::Mask:       field_variant_fx64<Variant::Mask>      (p, out); break;
        case Variant::Ship:       field_variant_fx64<Variant::Ship>      (p, out); break;
        case Variant::SinZ:       field_variant_fx64<Variant::SinZ>      (p, out); break;
        case Variant::CosZ:       field_variant_fx64<Variant::CosZ>      (p, out); break;
        case Variant::ExpZ:       field_variant_fx64<Variant::ExpZ>      (p, out); break;
        case Variant::SinhZ:      field_variant_fx64<Variant::SinhZ>     (p, out); break;
        case Variant::CoshZ:      field_variant_fx64<Variant::CoshZ>     (p, out); break;
        case Variant::TanZ:       field_variant_fx64<Variant::TanZ>      (p, out); break;
        default: break;  // Variant::Custom intercepted before this dispatch
    }
}

} // anonymous namespace (field kernels)

MapStats render_map_field(const MapParams& p, FieldOutput& fo) {
    // Custom variant: use function-pointer path (always OpenMP).
    if (p.variant == Variant::Custom && p.custom_step_fn) {
        return render_custom_field_openmp(p, fo);
    }

    fo.width  = p.width;
    fo.height = p.height;
    fo.metric = p.metric;

    const bool fx = use_fx64(p);
    const auto t0 = std::chrono::steady_clock::now();

    if (fx) dispatch_field_fx64(p, fo);
    else    dispatch_field_fp64(p, fo);

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = p.width * p.height;
    s.scalar_used = fx ? "fx64" : "fp64";
    s.engine_used = "openmp";
    fo.scalar_used = s.scalar_used;
    return s;
}

// ─── Colorized map renderer ───────────────────────────────────────────────────

MapStats render_map(const MapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    // Custom variant: bypass CUDA/AVX and go straight to OpenMP.
    if (p.variant == Variant::Custom && p.custom_step_fn) {
        return render_custom_openmp(p, out);
    }

    const bool fx  = use_fx64(p);

    // smooth coloring needs per-pixel |z|² which the AVX-512/CUDA paths don't track.
    // Fall through to OpenMP which has access to IterResult.norm.
    const bool needs_norm = p.smooth;

    // Trig variants need scalar (std::cmath) — skip AVX-512 and CUDA for them.
    const bool scalar_fallback = variant_needs_scalar_fallback(p.variant);

    // CUDA path: all 10 polynomial variants, Julia mode, metrics 0-3 (not MinPairwiseDist=4).
    // Trig variants fall to OpenMP (scalar_fallback).
    // smooth coloring (needs IterResult.norm) still falls to OpenMP.
#if USE_CUDA
    const bool can_cuda = !needs_norm
                       && !scalar_fallback
                       && (p.engine == "cuda" || p.engine == "auto" || p.engine == "hybrid")
                       && (static_cast<int>(p.metric) < 4)  // excludes MinPairwiseDist
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
        cp.variant_id   = static_cast<int>(p.variant);
        cp.julia        = p.julia;
        cp.julia_re     = p.julia_re;
        cp.julia_im     = p.julia_im;
        cp.metric_id    = static_cast<int>(p.metric);
        auto cs = fsd_cuda::cuda_render_map(cp, out);
        MapStats s;
        s.elapsed_ms  = cs.elapsed_ms;
        s.pixel_count = p.width * p.height;
        s.scalar_used = cs.scalar_used;
        s.engine_used = "cuda";
        return s;
    }
#endif

    // AVX-512 path: all 10 polynomial variants, Julia mode, metrics 0-3 (fp64).
    // For fx64 (IFMA52): Mandelbrot-only variant is supported; non-Mandelbrot
    // variants fall through to scalar OpenMP for correctness.
    // MinPairwiseDist (metric 4) is excluded: O(N²) orbit buffer not vectorised.
    // Smooth coloring needs per-pixel norm from IterResult — falls to OpenMP.
    // Trig variants need std::cmath — skip AVX-512 (scalar_fallback).
    const bool can_avx_base = !needs_norm
                           && !scalar_fallback
                           && (p.engine == "avx512" || p.engine == "auto" || p.engine == "hybrid")
                           && (static_cast<int>(p.metric) < 4)
                           && avx512_available();
    // fx64 path: restrict to Mandelbrot variant only (IFMA52 integer variant
    // extensions are not yet implemented for the other 9 variants).
    const bool can_avx = can_avx_base
                      && (!fx || p.variant == Variant::Mandelbrot);

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
