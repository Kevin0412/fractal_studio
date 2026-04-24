// compute/transition_volume.hpp
//
// Evaluate the 3D Mandelbrot↔Burning-Ship transition iteration on a dense
// grid and return the resulting scalar field (one float per voxel). The user
// then runs marching cubes on that field to get a 3D mesh of the set.
//
// Field values are normalized escape-time (0 = inside, 1 = escaped immediately),
// so the isosurface at `iso ≈ 0.5` gives the boundary of the set.

#pragma once

#include "marching_cubes.hpp"
#include "variants.hpp"

namespace fsd::compute {

struct TransitionVolumeParams {
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;

    // Half-extent (world units) in each axis around `center`. A cube by
    // default matching the Mandelbrot parameter bounds.
    double extent = 2.0;

    int resolution = 96;  // resolution³ voxels
    int iterations = 256;
    double bailout = 2.0;

    Variant from_variant = Variant::Mandelbrot;
    Variant to_variant   = Variant::Boat;
};

McField buildTransitionVolume(const TransitionVolumeParams& p);

} // namespace fsd::compute
