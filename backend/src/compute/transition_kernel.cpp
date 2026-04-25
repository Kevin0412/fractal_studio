// compute/transition_kernel.cpp

#include "transition_kernel.hpp"
#include "map_kernel.hpp"
#include "parallel.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fsd::compute {

namespace {

struct TransitionIterResult {
    int    iter;
    double min_abs_sq;
    double max_abs_sq;
    double extra;    // min pairwise distance (sqrt), or 0 if not computed
    double norm;     // |z|² at escape step (0 if not escaped)
    bool   escaped;
    IterResultMask valid_mask;
};

// Direct 3D transition iteration. Tracks min/max of x²+y²+z² for HS metrics.
// Matches cfiles/mandelbrot_3Dtranslation_minmax.c:52.
// When metric == MinPairwiseDist, also stores the orbit and computes min pairwise dist.
struct OrbitPt { double x, y, z; };

inline TransitionIterResult iterate_transition(
    double x0, double y0, double z0,
    int max_iter, double bail2,
    Variant from_variant, Variant to_variant,
    Metric metric, IterResultMask result_mask, int pairwise_cap,
    std::vector<OrbitPt>& orbit
) {
    (void)metric;
    double x = x0, y = y0, z = z0;
    TransitionIterResult r{};
    r.iter = 0;
    r.min_abs_sq = std::numeric_limits<double>::infinity();
    r.max_abs_sq = 0.0;
    r.extra = 0.0;
    r.norm = 0.0;
    r.escaped = false;
    r.valid_mask = 0;

    const bool track_iter    = iter_result_wants(result_mask, IterResultField::Iter);
    const bool track_min_abs = iter_result_wants(result_mask, IterResultField::MinAbs);
    const bool track_max_abs = iter_result_wants(result_mask, IterResultField::MaxAbs);
    const bool track_norm    = iter_result_wants(result_mask, IterResultField::Norm);
    const bool track_escaped = iter_result_wants(result_mask, IterResultField::Escaped);
    const bool track_orbit   = iter_result_wants(result_mask, IterResultField::Extra)
                            && pairwise_cap > 0;

    if (track_iter)    r.valid_mask |= IterResultField::Iter;
    if (track_min_abs) r.valid_mask |= IterResultField::MinAbs;
    if (track_max_abs) r.valid_mask |= IterResultField::MaxAbs;
    if (track_norm)    r.valid_mask |= IterResultField::Norm;
    if (track_escaped) r.valid_mask |= IterResultField::Escaped;
    if (track_orbit)   r.valid_mask |= IterResultField::Extra;

    if (track_min_abs || track_max_abs) {
        const double init_n2 = x*x + y*y + z*z;
        r.min_abs_sq = init_n2;
        r.max_abs_sq = init_n2;
    }

    if (track_orbit) {
        orbit.clear();
        orbit.push_back({x, y, z});
    }

    for (int i = 0; i < max_iter; i++) {
        const double x2 = x * x;
        const double nx =
            variant_transition_real_projection(from_variant, x2, y * y)
          + variant_transition_real_projection(to_variant,   x2, z * z)
          - x2 + x0;
        const double ny = variant_transition_imag_projection(from_variant, x, y) + y0;
        const double nz = variant_transition_imag_projection(to_variant,   x, z) + z0;
        x = nx; y = ny; z = nz;
        const bool finite_xyz = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
        const double n2 = finite_xyz
            ? (x*x + y*y + z*z)
            : std::numeric_limits<double>::infinity();
        if (track_min_abs && n2 < r.min_abs_sq) r.min_abs_sq = n2;
        if (track_max_abs && n2 > r.max_abs_sq) r.max_abs_sq = n2;

        if (track_orbit && static_cast<int>(orbit.size()) < pairwise_cap) {
            orbit.push_back({x, y, z});
        }

        if (!finite_xyz || n2 > bail2) {
            if (track_iter)    r.iter = i;
            if (track_norm)    r.norm = n2;
            if (track_escaped) r.escaped = true;
            break;
        }
    }
    if (!r.escaped) {
        if (track_iter) r.iter = max_iter;
    }

    // Compute min pairwise distance from orbit (O(n²), capped).
    if (track_orbit && orbit.size() >= 2) {
        double min_d2 = std::numeric_limits<double>::max();
        for (size_t a = 0; a < orbit.size(); a++) {
            for (size_t b = a + 1; b < orbit.size(); b++) {
                const double dx = orbit[a].x - orbit[b].x;
                const double dy = orbit[a].y - orbit[b].y;
                const double dz = orbit[a].z - orbit[b].z;
                const double d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < min_d2) min_d2 = d2;
            }
        }
        r.extra = std::sqrt(min_d2);
    }

    return r;
}

} // namespace

MapStats render_transition(const TransitionParams& p, cv::Mat& out) {
    if (!variant_supports_axis_transition(p.from_variant) ||
        !variant_supports_axis_transition(p.to_variant)) {
        throw std::runtime_error("transition variants must be quadratic Mandelbrot-family variants");
    }

    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    const auto t0 = std::chrono::steady_clock::now();

    const int W = p.width;
    const int H = p.height;
    const double aspect  = static_cast<double>(W) / H;
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout_sq;
    const double cth     = std::cos(p.theta);
    const double sth     = std::sin(p.theta);
    const IterResultMask result_mask =
        iter_result_mask_for_metric(p.metric, p.metric == Metric::Escape && p.smooth);
    const int thread_count = resolve_render_threads(p.render_threads);

    #pragma omp parallel num_threads(thread_count)
    {
        std::vector<OrbitPt> orbit;
        orbit.reserve(static_cast<size_t>(p.pairwise_cap));

    #pragma omp for schedule(dynamic, 4)
    for (int y = 0; y < H; y++) {
        uint8_t* row = out.ptr<uint8_t>(y);
        const double v = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
        for (int x = 0; x < W; x++) {
            const double u = re_min + (static_cast<double>(x) + 0.5) / W * span_re;

            const double x0 = u;
            const double y0 = v * cth;
            const double z0 = v * sth;

            const TransitionIterResult r =
                iterate_transition(x0, y0, z0, p.iterations, bail2,
                                   p.from_variant, p.to_variant,
                                   p.metric, result_mask, p.pairwise_cap, orbit);

            uint8_t* px = row + 3 * x;
            if (p.metric == Metric::Escape) {
                const int    iter = r.escaped ? r.iter : p.iterations;
                const double norm = r.escaped ? r.norm : 0.0;
                colorize_escape_bgr(iter, p.iterations, p.colormap, norm, p.smooth, px[0], px[1], px[2]);
            } else if (p.colormap == Colormap::HsRainbow) {
                double fv = 0.0;
                if (p.metric == Metric::MinAbs)
                    fv = std::sqrt(r.min_abs_sq);
                else if (p.metric == Metric::MaxAbs)
                    fv = std::sqrt(r.max_abs_sq);
                else if (p.metric == Metric::Envelope)
                    fv = 0.5 * (std::sqrt(r.min_abs_sq) + std::sqrt(r.max_abs_sq));
                else if (p.metric == Metric::MinPairwiseDist)
                    fv = r.extra;
                colorize_field_hs_bgr(fv, px[0], px[1], px[2]);
            } else if (p.smooth) {
                double fv = 0.0;
                if (p.metric == Metric::MinAbs)
                    fv = std::sqrt(r.min_abs_sq);
                else if (p.metric == Metric::MaxAbs)
                    fv = std::sqrt(r.max_abs_sq);
                else if (p.metric == Metric::Envelope)
                    fv = 0.5 * (std::sqrt(r.min_abs_sq) + std::sqrt(r.max_abs_sq));
                else if (p.metric == Metric::MinPairwiseDist)
                    fv = r.extra;
                colorize_field_smooth_bgr(fv, p.colormap, px[0], px[1], px[2]);
            } else {
                double v01 = 0.0;
                if (p.metric == Metric::MinAbs)
                    v01 = std::min(1.0, std::sqrt(r.min_abs_sq) / p.bailout);
                else if (p.metric == Metric::MaxAbs)
                    v01 = std::min(1.0, std::sqrt(r.max_abs_sq) / p.bailout);
                else if (p.metric == Metric::Envelope)
                    v01 = std::min(1.0, 0.5 * (std::sqrt(r.min_abs_sq) + std::sqrt(r.max_abs_sq)) / p.bailout);
                else if (p.metric == Metric::MinPairwiseDist)
                    v01 = std::min(1.0, r.extra / p.bailout);
                colorize_field_bgr(v01, p.colormap, px[0], px[1], px[2]);
            }
        }
    }
    } // end omp parallel

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = p.width * p.height;
    return s;
}

} // namespace fsd::compute
