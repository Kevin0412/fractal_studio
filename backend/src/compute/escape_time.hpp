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

// Per-pixel result. Fields are populated conditionally on metric.
struct IterResult {
    int    iter;     // escape iteration (== max_iter if bounded)
    double min_abs;  // min |z_n|, valid for MinAbs/Envelope/MinPairwiseDist fallback
    double max_abs;  // max |z_n|, valid for MaxAbs/Envelope
    double extra;    // MinPairwiseDist result
    double norm;     // |z|² at the escape step (0 if not escaped); used by LnSmooth
    bool   escaped;
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
    int pairwise_cap,
    std::vector<Cx<S>>& orbit_scratch  // reused across pixels by caller
) {
    IterResult r{};
    r.iter    = 0;
    r.min_abs = std::numeric_limits<double>::infinity();
    r.max_abs = 0.0;
    r.extra   = std::numeric_limits<double>::infinity();
    r.norm    = 0.0;
    r.escaped = false;

    const bool track_pairwise = (metric == Metric::MinPairwiseDist);
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

        if (metric == Metric::MinAbs || metric == Metric::Envelope || metric == Metric::MinPairwiseDist) {
            if (n2 < r.min_abs) r.min_abs = n2;
        }
        if (metric == Metric::MaxAbs || metric == Metric::Envelope) {
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
            r.iter = i;
            r.norm = n2;
            r.escaped = true;
            // Convert squared accumulators to magnitudes on exit.
            if (r.min_abs != std::numeric_limits<double>::infinity()) r.min_abs = scalar_sqrt(r.min_abs);
            if (r.max_abs != 0.0)                                     r.max_abs = scalar_sqrt(r.max_abs);
            if (r.extra   != std::numeric_limits<double>::infinity()) r.extra   = scalar_sqrt(r.extra);
            return r;
        }
    }

    r.iter    = n_iter;
    r.escaped = false;
    if (r.min_abs != std::numeric_limits<double>::infinity()) r.min_abs = scalar_sqrt(r.min_abs);
    if (r.max_abs != 0.0)                                     r.max_abs = scalar_sqrt(r.max_abs);
    if (r.extra   != std::numeric_limits<double>::infinity()) r.extra   = scalar_sqrt(r.extra);
    return r;
}

} // namespace fsd::compute
