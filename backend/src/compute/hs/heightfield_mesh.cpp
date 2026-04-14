// compute/hs/heightfield_mesh.cpp

#include "heightfield_mesh.hpp"
#include "../complex.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fsd::compute::hs {

namespace {

// Populate `field` with per-pixel metric values using the same iterate<>
// core as the map kernel. Field is row-major: field[row * N + col].
template <Variant V>
void computeFieldImpl(const HsMeshParams& p, std::vector<double>& field) {
    const int N = p.resolution;
    field.assign(N * N, 0.0);
    const double span = p.scale;
    const double re_min = p.center_re - span * 0.5;
    const double im_max = p.center_im + span * 0.5;
    const double bail2 = p.bailout * p.bailout;

    #pragma omp parallel
    {
        std::vector<Cx<double>> orbit_scratch;
        orbit_scratch.reserve(64);

        #pragma omp for schedule(dynamic, 4)
        for (int row = 0; row < N; row++) {
            const double im = im_max - (static_cast<double>(row) + 0.5) / N * span;
            for (int col = 0; col < N; col++) {
                const double re = re_min + (static_cast<double>(col) + 0.5) / N * span;
                Cx<double> c{re, im};
                Cx<double> z0{0.0, 0.0};
                IterResult r = iterate<V, double>(
                    z0, c, p.iterations, bail2, p.metric, 64, orbit_scratch);

                double v = 0.0;
                switch (p.metric) {
                    case Metric::Escape:
                        v = r.escaped
                            ? static_cast<double>(r.iter) / static_cast<double>(p.iterations)
                            : 1.0;
                        break;
                    case Metric::MinAbs:          v = std::isfinite(r.min_abs) ? r.min_abs : p.heightClamp; break;
                    case Metric::MaxAbs:          v = r.max_abs; break;
                    case Metric::Envelope:        v = 0.5 * (r.min_abs + r.max_abs); break;
                    case Metric::MinPairwiseDist: v = std::isfinite(r.extra) ? r.extra : p.heightClamp; break;
                }
                if (v > p.heightClamp) v = p.heightClamp;
                if (!std::isfinite(v)) v = p.heightClamp;
                field[row * N + col] = v;
            }
        }
    }
}

} // namespace

// Public: compute raw metric field. Used by both buildHsMesh and hsFieldRoute.
void computeHsField(const HsMeshParams& p, std::vector<double>& field) {
    using V = Variant;
    switch (p.variant) {
        case V::Mandelbrot: computeFieldImpl<V::Mandelbrot>(p, field); break;
        case V::Tri:        computeFieldImpl<V::Tri>       (p, field); break;
        case V::Boat:       computeFieldImpl<V::Boat>      (p, field); break;
        case V::Duck:       computeFieldImpl<V::Duck>      (p, field); break;
        case V::Bell:       computeFieldImpl<V::Bell>      (p, field); break;
        case V::Fish:       computeFieldImpl<V::Fish>      (p, field); break;
        case V::Vase:       computeFieldImpl<V::Vase>      (p, field); break;
        case V::Bird:       computeFieldImpl<V::Bird>      (p, field); break;
        case V::Mask:       computeFieldImpl<V::Mask>      (p, field); break;
        case V::Ship:       computeFieldImpl<V::Ship>      (p, field); break;
        case V::SinZ:       computeFieldImpl<V::SinZ>      (p, field); break;
        case V::CosZ:       computeFieldImpl<V::CosZ>      (p, field); break;
        case V::ExpZ:       computeFieldImpl<V::ExpZ>      (p, field); break;
        case V::SinhZ:      computeFieldImpl<V::SinhZ>     (p, field); break;
        case V::CoshZ:      computeFieldImpl<V::CoshZ>     (p, field); break;
        case V::TanZ:       computeFieldImpl<V::TanZ>      (p, field); break;
        default:            computeFieldImpl<V::Mandelbrot>(p, field); break;
    }
}

Mesh buildHsMesh(const HsMeshParams& p) {
    const int N = std::max(4, std::min(4096, p.resolution));

    std::vector<double> field;
    HsMeshParams pp = p;
    pp.resolution = N;
    computeHsField(pp, field);

    // Normalize field to [0, 1] for visual consistency.
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double v : field) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const double denom = (hi > lo) ? (hi - lo) : 1.0;

    Mesh mesh;
    mesh.vertices.resize(static_cast<size_t>(N) * static_cast<size_t>(N));
    mesh.indices.reserve(static_cast<size_t>((N - 1) * (N - 1) * 6));

    const double heightScale = p.heightScale;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            const double u = static_cast<double>(col) / static_cast<double>(N - 1) - 0.5;
            const double v = static_cast<double>(row) / static_cast<double>(N - 1) - 0.5;
            const double f01 = (field[row * N + col] - lo) / denom;
            mesh.vertices[row * N + col] = Vec3{
                static_cast<float>(u),
                static_cast<float>(-v),  // flip so row 0 is +Y (image convention)
                static_cast<float>(f01 * heightScale),
            };
        }
    }

    for (int row = 0; row < N - 1; row++) {
        for (int col = 0; col < N - 1; col++) {
            const uint32_t a = static_cast<uint32_t>(row       * N + col);
            const uint32_t b = static_cast<uint32_t>(row       * N + col + 1);
            const uint32_t c = static_cast<uint32_t>((row + 1) * N + col);
            const uint32_t d = static_cast<uint32_t>((row + 1) * N + col + 1);
            // Two triangles per cell, consistent winding (CCW when viewed
            // from +Z).
            mesh.indices.push_back(a);
            mesh.indices.push_back(c);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
            mesh.indices.push_back(d);
        }
    }

    return mesh;
}

} // namespace fsd::compute::hs
