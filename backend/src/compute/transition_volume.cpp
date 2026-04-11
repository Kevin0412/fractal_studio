// compute/transition_volume.cpp

#include "transition_volume.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <cmath>

namespace fsd::compute {

McField buildTransitionVolume(const TransitionVolumeParams& p) {
    const int N = std::max(4, std::min(512, p.resolution));
    McField field;
    field.Nx = field.Ny = field.Nz = N;
    field.data.assign(static_cast<size_t>(N) * N * N, 1.0f);

    const double span = p.extent * 2.0;
    const double xmin = p.centerX - p.extent;
    const double ymin = p.centerY - p.extent;
    const double zmin = p.centerZ - p.extent;
    const double bail2 = p.bailout * p.bailout;

    const int maxIter = p.iterations;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int zi = 0; zi < N; zi++) {
        const double z0 = zmin + (zi + 0.5) / N * span;
        for (int yi = 0; yi < N; yi++) {
            const double y0 = ymin + (yi + 0.5) / N * span;
            for (int xi = 0; xi < N; xi++) {
                const double x0 = xmin + (xi + 0.5) / N * span;

                double x = x0, y = y0, z = z0;
                int iter = 0;
                bool escaped = false;
                for (; iter < maxIter; iter++) {
                    const double nx = x*x - y*y - z*z + x0;
                    const double ny = 2.0 * x * y + y0;
                    const double nz = 2.0 * std::fabs(x * z) + z0;
                    x = nx; y = ny; z = nz;
                    if (x*x + y*y + z*z > bail2) {
                        escaped = true;
                        break;
                    }
                }

                // Smooth escape-time → scalar in [0, 1]. Inside points = 0,
                // outside = fraction of iterations before escape.
                float v = 0.0f;
                if (escaped) {
                    v = static_cast<float>(iter) / static_cast<float>(maxIter);
                    v = 0.5f + 0.5f * v;  // push outside above the iso level
                } else {
                    v = 0.0f;  // inside → below iso
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
