// compute/cuda/map_kernel.cuh
//
// Host-side interface to the CUDA map renderer.
// Includes fp64 and fx64 variants for all 10 fractal variants,
// Julia mode, and non-escape metrics (MinAbs, MaxAbs, Envelope).
// CUDA Graphs are used to minimise kernel-launch overhead when streaming tiles.

#pragma once

#include <cstdint>
#include <string>

// Forward-declare OpenCV types without pulling in all headers.
namespace cv { class Mat; }

namespace fsd_cuda {

struct CudaMapParams {
    double center_re = -0.75;
    double center_im =  0.0;
    double scale     =  3.0;

    int width       = 1024;
    int height      = 768;
    int iterations  = 1024;
    double bailout  = 2.0;  // radius, kept for metric normalization
    double bailout_sq = 4.0; // squared threshold used by escape tests

    // "fp64" or "fx64"
    std::string scalar_type = "fp64";

    // Colormap ID — must match fsd::compute::Colormap enum values:
    //   0=ClassicCos, 1=Mod17, 2=HsvWheel, 3=Tri765, 4=Grayscale
    //   (LnSmooth=5 is never passed here; handled by CPU path)
    int colormap_id = 0;

    // Variant ID — matches fsd::compute::Variant enum:
    //   0=Mandelbrot, 1=Tri, 2=Boat, 3=Duck, 4=Bell,
    //   5=Fish, 6=Vase, 7=Bird, 8=Mask, 9=Ship
    int variant_id = 0;

    // Julia mode: if true, z0 = pixel, c = (julia_re, julia_im)
    bool julia    = false;
    double julia_re = 0.0;
    double julia_im = 0.0;

    // Metric ID — matches fsd::compute::Metric enum:
    //   0=Escape, 1=MinAbs, 2=MaxAbs, 3=Envelope
    //   (MinPairwiseDist=4 is NOT supported on CUDA — use CPU path)
    int metric_id = 0;
};

struct CudaMapStats {
    double elapsed_ms = 0.0;
    std::string scalar_used;
    std::string engine_used = "cuda";
};

// Returns true if a CUDA device is present and initialised.
bool cuda_available() noexcept;

// Render a fractal map using the CUDA kernels.
// Output `out` is allocated as CV_8UC3 BGR on the host.
// Internally uses CUDA Graphs to amortise launch overhead.
CudaMapStats cuda_render_map(const CudaMapParams& p, cv::Mat& out);

} // namespace fsd_cuda
