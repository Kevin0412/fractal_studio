// compute/escape_time.hpp
//
// Per-pixel iteration core. Given a point `c` (or for Julia, a fixed `c` and
// a variable starting `z`), iterate the selected variant up to `max_iter`
// steps or until escape, and accumulate the requested metric.
//
// Metrics:
//   Escape          classic escape-time (iteration index at which |z|>2)
//   MinAbs          min_{n≤N} |z_n|              — HS-Base field
//   MaxAbs          max_{n≤N} |z_n|
//   Envelope        returns both MinAbs and MaxAbs packed into one sample
//   MinPairwiseDist min_{i<j} |z_i − z_j|        — HS-Recurrence field
//                   (O(N²) per pixel; kernel caps N for this metric)
//
// The returned `IterResult` is the common shape for all metrics. Kernels and
// colormap code read whichever fields their metric populated.

#pragma once

#include "complex.hpp"
#include "variants.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

namespace fsd::compute {

enum class Metric {
    Escape          = 0,
    MinAbs          = 1,
    MaxAbs          = 2,
    Envelope        = 3,
    MinPairwiseDist = 4,
};

inline const char* metric_name(Metric m) {
    switch (m) {
        case Metric::Escape:          return "escape";
        case Metric::MinAbs:          return "min_abs";
        case Metric::MaxAbs:          return "max_abs";
        case Metric::Envelope:        return "envelope";
        case Metric::MinPairwiseDist: return "min_pairwise_dist";
    }
    return "escape";
}

inline bool metric_from_name(const char* name, Metric& out) {
    struct Entry { const char* n; Metric m; };
    static constexpr Entry table[] = {
        {"escape",            Metric::Escape},
        {"min_abs",           Metric::MinAbs},
        {"max_abs",           Metric::MaxAbs},
        {"envelope",          Metric::Envelope},
        {"min_pairwise_dist", Metric::MinPairwiseDist},
    };
    for (const auto& e : table) {
        const char* a = e.n;
        const char* b = name;
        while (*a && *b && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) { out = e.m; return true; }
    }
    return false;
}

using IterResultMask = uint32_t;

namespace IterResultField {
    inline constexpr IterResultMask Iter     = 1u << 0;
    inline constexpr IterResultMask MinAbs   = 1u << 1;
    inline constexpr IterResultMask MaxAbs   = 1u << 2;
    inline constexpr IterResultMask Extra    = 1u << 3;
    inline constexpr IterResultMask Norm     = 1u << 4;
    inline constexpr IterResultMask Escaped  = 1u << 5;
}

inline bool iter_result_wants(IterResultMask mask, IterResultMask field) {
    return (mask & field) != 0;
}

inline IterResultMask iter_result_mask_for_metric(Metric metric, bool need_escape_norm = false) {
    switch (metric) {
        case Metric::Escape:
            return IterResultField::Iter
                 | IterResultField::Escaped
                 | (need_escape_norm ? IterResultField::Norm : 0u);
        case Metric::MinAbs:
            return IterResultField::MinAbs;
        case Metric::MaxAbs:
            return IterResultField::MaxAbs;
        case Metric::Envelope:
            return IterResultField::MinAbs | IterResultField::MaxAbs;
        case Metric::MinPairwiseDist:
            return IterResultField::Extra;
    }
    return IterResultField::Iter | IterResultField::Escaped;
}

// Per-pixel result. `valid_mask` reports which fields were requested and
// maintained by the caller-selected mask.
struct IterResult {
    int    iter;     // escape iteration (== max_iter if bounded)
    double min_abs;  // min |z_n|, valid when valid_mask has MinAbs
    double max_abs;  // max |z_n|, valid for MaxAbs/Envelope
    double extra;    // MinPairwiseDist result
    double norm;     // |z|² at the escape step (0 if not escaped); used by LnSmooth
    bool   escaped;
    IterResultMask valid_mask;
};

// Iterate up to max_iter steps of variant V on seed (z0, c). Tracks metrics.
// For MinPairwiseDist we keep an orbit buffer capped at `pairwise_cap`.
template <Variant V, typename S>
inline IterResult iterate(
    Cx<S> z,
    const Cx<S>& c,
    int max_iter,
    double bailout,
    double bailout_sq,
    Metric metric,
    IterResultMask result_mask,
    int pairwise_cap,
    std::vector<Cx<S>>& orbit_scratch  // reused across pixels by caller
) {
    (void)metric;
    IterResult r{};
    r.iter    = 0;
    r.min_abs = std::numeric_limits<double>::infinity();
    r.max_abs = 0.0;
    r.extra   = std::numeric_limits<double>::infinity();
    r.norm    = 0.0;
    r.escaped = false;
    r.valid_mask = 0;

    const bool track_iter     = iter_result_wants(result_mask, IterResultField::Iter);
    const bool track_min_abs  = iter_result_wants(result_mask, IterResultField::MinAbs);
    const bool track_max_abs  = iter_result_wants(result_mask, IterResultField::MaxAbs);
    const bool track_pairwise = iter_result_wants(result_mask, IterResultField::Extra)
                             && pairwise_cap > 0;
    const bool track_norm     = iter_result_wants(result_mask, IterResultField::Norm);
    const bool track_escaped  = iter_result_wants(result_mask, IterResultField::Escaped);

    if (track_iter)     r.valid_mask |= IterResultField::Iter;
    if (track_min_abs)  r.valid_mask |= IterResultField::MinAbs;
    if (track_max_abs)  r.valid_mask |= IterResultField::MaxAbs;
    if (track_pairwise) r.valid_mask |= IterResultField::Extra;
    if (track_norm)     r.valid_mask |= IterResultField::Norm;
    if (track_escaped)  r.valid_mask |= IterResultField::Escaped;

    if (track_pairwise) {
        orbit_scratch.clear();
        orbit_scratch.reserve(static_cast<size_t>(pairwise_cap));
    }

    int n_iter = (track_pairwise && max_iter > pairwise_cap) ? pairwise_cap : max_iter;

    int i;
    for (i = 0; i < n_iter; i++) {
        z = variant_step<V, S>(z, c);
        // Compute |z|² in double space: avoids fixed-point overflow when |z|
        // grows large near the escape boundary (re² + im² can exceed Fx64 range).
        const double zre = scalar_to_double(z.re);
        const double zim = scalar_to_double(z.im);
        const bool finite_z = std::isfinite(zre) && std::isfinite(zim);
        const double n2  = finite_z
            ? (zre * zre + zim * zim)
            : std::numeric_limits<double>::infinity();

        if (track_min_abs) {
            if (n2 < r.min_abs) r.min_abs = n2;
        }
        if (track_max_abs) {
            if (n2 > r.max_abs) r.max_abs = n2;
        }
        if (track_pairwise) {
            // Compare new point against all prior orbit points.
            for (const auto& prior : orbit_scratch) {
                const double dr = zre - scalar_to_double(prior.re);
                const double di = zim - scalar_to_double(prior.im);
                const double d2 = dr * dr + di * di;
                if (d2 < r.extra) r.extra = d2;
            }
            orbit_scratch.push_back(z);
        }

        bool escaped_now = !finite_z;
        if constexpr (variant_is_transcendental_v<V>()) {
            const double component = std::max(std::fabs(zre), std::fabs(zim));
            escaped_now = escaped_now || component >= bailout;
        } else {
            escaped_now = escaped_now || n2 > bailout_sq;
        }

        if (escaped_now) {
            if (track_iter)    r.iter = i;
            if (track_norm)    r.norm = n2;
            if (track_escaped) r.escaped = true;
            // Convert squared accumulators to magnitudes on exit.
            if (track_min_abs && r.min_abs != std::numeric_limits<double>::infinity()) r.min_abs = scalar_sqrt(r.min_abs);
            if (track_max_abs && r.max_abs != 0.0)                                     r.max_abs = scalar_sqrt(r.max_abs);
            if (track_pairwise && r.extra != std::numeric_limits<double>::infinity())   r.extra   = scalar_sqrt(r.extra);
            return r;
        }
    }

    if (track_iter)    r.iter = n_iter;
    if (track_escaped) r.escaped = false;
    if (track_min_abs && r.min_abs != std::numeric_limits<double>::infinity()) r.min_abs = scalar_sqrt(r.min_abs);
    if (track_max_abs && r.max_abs != 0.0)                                     r.max_abs = scalar_sqrt(r.max_abs);
    if (track_pairwise && r.extra != std::numeric_limits<double>::infinity())   r.extra   = scalar_sqrt(r.extra);
    return r;
}

} // namespace fsd::compute
