// compute/transition_kernel.hpp
//
// User's innovation: the continuous 3D bridge between Mandelbrot and Burning
// Ship dynamics (README.md:39).
//
//   x_{n+1} = x² − y² − z² + x₀
//   y_{n+1} = 2·x·y     + y₀
//   z_{n+1} = 2·|x·z|   + z₀
//
// In the xy-plane (z=0) the iteration is exactly Mandelbrot. In the xz-plane
// (y=0) the iteration is Burning-Ship-like (the |·| on the xz cross-term is
// what folds the imaginary axis and produces the ship hull).
//
// A 2D map at rotation angle θ around the x-axis embeds the screen's
// imaginary axis into 3D as (cosθ·y + sinθ·z). Concretely a pixel (u, v)
// maps to seed (x₀, y₀, z₀) = (u, v·cosθ, v·sinθ). θ=0 → Mandelbrot,
// θ=π/2 → Burning Ship, anything in between is the continuous bridge.
//
// This matches cfiles/mandelbrot_3Dtranslation_minmax.c exactly (see its
// `mandelbrot(c, max, num, theta)` at line 52).

#pragma once

#include "colormap.hpp"
#include "escape_time.hpp"
#include "map_kernel.hpp"  // for MapStats
#include "variants.hpp"

#include <opencv2/core.hpp>

namespace fsd::compute {

struct TransitionParams {
    double center_re = -0.5;
    double center_im =  0.0;
    double scale     =  4.0;

    int width        = 1024;
    int height       = 1024;
    int iterations   = 512;
    double bailout   = 2.0;

    // Rotation angle around x-axis, radians. 0 = Mandelbrot, π/2 = Burning Ship.
    double theta     = 0.0;

    // Quadratic/folded variants to place on the xy and xz planes. The default
    // preserves the original Mandelbrot → Burning Ship bridge.
    Variant from_variant = Variant::Mandelbrot;
    Variant to_variant   = Variant::Boat;

    Metric   metric       = Metric::Escape;
    Colormap colormap     = Colormap::ClassicCos;
    bool     smooth       = false;
    int      pairwise_cap = 64;   // orbit length cap for MinPairwiseDist
};

MapStats render_transition(const TransitionParams& p, cv::Mat& out);

} // namespace fsd::compute
