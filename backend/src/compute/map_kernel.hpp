// compute/map_kernel.hpp
//
// Public entry for rendering a 2D map (Mandelbrot-family variant or Julia
// subset) into an OpenCV BGR Mat. All combinations of variant × metric are
// dispatched here. Transition (3D-rotated) slices live in transition_kernel.

#pragma once

#include "colormap.hpp"
#include "escape_time.hpp"
#include "variants.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace fsd::compute {

struct MapParams {
    // Center in parameter space and axis-aligned scale:
    //   - `scale` is the HEIGHT of the viewport in complex units.
    //   - width span is scale * width/height.
    double center_re = -0.75;
    double center_im =  0.0;
    double scale     =  3.0;

    int width        = 1024;
    int height       = 768;
    int iterations   = 1024;
    double bailout   = 2.0;

    Variant  variant  = Variant::Mandelbrot;
    Metric   metric   = Metric::Escape;
    Colormap colormap = Colormap::ClassicCos;
    bool     smooth   = false;   // ln-smooth continuous coloring (requires norm)

    // Julia mode: if true, seed z = pixel point, c = (julia_re, julia_im).
    bool julia        = false;
    double julia_re   = 0.0;
    double julia_im   = 0.0;

    // Cap for the pairwise-distance orbit buffer (HS-Recurrence).
    int pairwise_cap  = 64;

    // Scalar type: "fp64" (default), "fx64", or "auto" (auto-selects fx64
    // when scale < 1e-13 for precision at extreme zoom depth).
    std::string scalar_type = "auto";

    // Compute engine: "openmp" (default), "avx512" (AVX-512 CPU kernel),
    // "cuda" (GPU kernel, Phase 3.3). Silently falls back to openmp if
    // requested engine is unavailable.
    std::string engine = "openmp";
};

struct MapStats {
    double elapsed_ms   = 0.0;
    int    pixel_count  = 0;
    std::string scalar_used;  // "fp64" or "fx64"
    std::string engine_used;  // "openmp", "avx512", etc.
};

// Render a map into `out` (allocated BGR CV_8UC3 of size height x width).
// Returns stats including which scalar type and engine were actually used.
MapStats render_map(const MapParams& p, cv::Mat& out);

} // namespace fsd::compute
