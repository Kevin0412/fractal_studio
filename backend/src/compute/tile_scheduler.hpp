// compute/tile_scheduler.hpp
//
// Hybrid CPU+GPU tile scheduler.
//
// A viewport is split into tiles (default 256×256). Tiles are pushed into a
// shared work queue. Two worker pools draw from the queue:
//
//   CPU pool — one worker per omp_get_max_threads() / 2 logical cores.
//              Runs the AVX-512 or OpenMP path depending on availability.
//
//   GPU pool — one worker. Runs CUDA batches of tiles via cudaGraphLaunch.
//              (Only present when HAS_CUDA_KERNEL is defined.)
//
// Assignment policy (EMA throughput):
//   Each worker tracks an EMA of pixels/sec. The scheduler dispatches the
//   next available tile to whichever pool currently has the highest EMA
//   throughput per tile-capacity unit.
//
// The full viewport render completes once all tiles are consumed and all
// workers have joined. Results are written directly into a pre-allocated
// output cv::Mat.
//
// For the HTTP route, the scheduler is called synchronously (the HTTP handler
// waits for completion). Streaming preview (Phase 4) would push partial
// results via a callback — the interface is intentionally callback-ready.

#pragma once

#include "map_kernel.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace fsd::compute {

struct TileSchedulerStats {
    double total_ms   = 0.0;
    double cpu_ms     = 0.0;
    double gpu_ms     = 0.0;
    int    cpu_tiles  = 0;
    int    gpu_tiles  = 0;
    std::string scalar_used;
    std::string engine_used;
};

// Optional per-tile completion callback (for streaming preview).
// Arguments: (tile_x, tile_y, tile_w, tile_h, BGR rows in out).
using TileCallback = std::function<void(int tx, int ty, int tw, int th)>;

// Render the full viewport described by `p` using a hybrid tile scheduler.
// The output mat `out` must be pre-allocated to p.width × p.height × CV_8UC3.
// `tile_size` controls the subdivision granularity.
TileSchedulerStats render_map_hybrid(
    const MapParams& p,
    cv::Mat& out,
    int tile_size = 256,
    TileCallback on_tile_done = nullptr
);

} // namespace fsd::compute
