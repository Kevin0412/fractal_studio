// compute/tile_scheduler.cpp
//
// Hybrid CPU+GPU tile scheduler implementation.

#include "tile_scheduler.hpp"
#include "map_kernel.hpp"
#include "map_kernel_avx512.hpp"
#include "parallel.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/map_kernel.cuh"
#  define USE_CUDA 1
#else
#  define USE_CUDA 0
#endif

#include <opencv2/core.hpp>
#ifdef _OPENMP
#  include <omp.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

namespace fsd::compute {

// ---- Tile descriptor ----

struct Tile {
    int x, y;     // top-left pixel in the full viewport
    int w, h;     // tile dimensions (may be smaller at right/bottom edges)
};

static std::vector<Tile> make_tiles(int W, int H, int tile_size) {
    std::vector<Tile> tiles;
    for (int ty = 0; ty < H; ty += tile_size) {
        for (int tx = 0; tx < W; tx += tile_size) {
            Tile t;
            t.x = tx;
            t.y = ty;
            t.w = std::min(tile_size, W - tx);
            t.h = std::min(tile_size, H - ty);
            tiles.push_back(t);
        }
    }
    return tiles;
}

// ---- Worker EMA throughput tracker ----

struct WorkerStats {
    double ema_pps = 0.0;   // EMA of pixels/sec
    double alpha   = 0.3;   // smoothing factor
    int    count   = 0;     // tiles processed

    void record(int pixels, double ms) {
        if (ms <= 0.0) return;
        const double pps = pixels / (ms / 1000.0);
        if (count == 0) {
            ema_pps = pps;
        } else {
            ema_pps = alpha * pps + (1.0 - alpha) * ema_pps;
        }
        count++;
    }
};

// ---- Tile render helpers ----

// Render a single tile into the output mat using the best available CPU path.
static double render_tile_cpu(
    const MapParams& base, const Tile& t,
    cv::Mat& out, bool use_avx, bool use_fx
) {
    // Build a MapParams for this tile.
    MapParams p = base;
    const double aspect  = static_cast<double>(base.width) / base.height;
    const double span_im = base.scale;
    const double span_re = base.scale * aspect;
    const double re_step = span_re / base.width;
    const double im_step = span_im / base.height;

    // Tile center_re/center_im: re/im at the tile's top-left pixel center,
    // then re-center for the tile viewport.
    const double re_min = (base.center_re - span_re * 0.5) + t.x * re_step;
    const double im_max = (base.center_im + span_im * 0.5) - t.y * im_step;

    p.center_re = re_min + (t.w * 0.5) * re_step;
    p.center_im = im_max - (t.h * 0.5) * im_step;
    p.scale     = t.h * im_step;    // height of this tile in complex units
    p.width     = t.w;
    p.height    = t.h;
    p.engine    = use_avx ? "avx512" : "openmp";
    p.scalar_type = use_fx ? "fx64" : "fp64";
    p.render_threads = 1;

    // Create a submat view for this tile.
    cv::Mat tile_mat = out(cv::Rect(t.x, t.y, t.w, t.h));

    MapStats stats;
    if (use_avx && p.variant == Variant::Mandelbrot && p.metric == Metric::Escape) {
        stats = use_fx ? render_map_avx512_fx64(p, tile_mat)
                       : render_map_avx512_fp64(p, tile_mat);
    } else {
        // OpenMP path (single-threaded for this tile since the outer scheduler
        // calls render_tile_cpu from multiple CPU workers concurrently).
        stats = render_map(p, tile_mat);
    }
    return stats.elapsed_ms;
}

#if USE_CUDA
static double render_tile_gpu(
    const MapParams& base, const Tile& t,
    cv::Mat& out, bool use_fx
) {
    MapParams p = base;
    const double aspect  = static_cast<double>(base.width) / base.height;
    const double span_re = base.scale * aspect;
    const double span_im = base.scale;
    const double re_step = span_re / base.width;
    const double im_step = span_im / base.height;
    const double re_min  = (base.center_re - span_re * 0.5) + t.x * re_step;
    const double im_max  = (base.center_im + span_im * 0.5) - t.y * im_step;

    fsd_cuda::CudaMapParams cp;
    cp.center_re  = re_min + (t.w * 0.5) * re_step;
    cp.center_im  = im_max - (t.h * 0.5) * im_step;
    cp.scale      = t.h * im_step;
    cp.width      = t.w;
    cp.height     = t.h;
    cp.iterations = base.iterations;
    cp.bailout    = base.bailout;
    cp.bailout_sq = base.bailout_sq;
    cp.scalar_type  = use_fx ? "fx64" : "fp64";
    cp.colormap_id  = static_cast<int>(base.colormap);
    cp.variant_id   = static_cast<int>(base.variant);
    cp.julia        = base.julia;
    cp.julia_re     = base.julia_re;
    cp.julia_im     = base.julia_im;
    cp.metric_id    = static_cast<int>(base.metric);

    // cuda_render_map does cudaMemcpy into a contiguous buffer; passing a submat
    // (whose rows are separated by the full-image stride) would corrupt adjacent
    // tiles.  Render into a fresh contiguous Mat and then blit into the submat.
    cv::Mat tile_buf;
    auto stats = fsd_cuda::cuda_render_map(cp, tile_buf);
    tile_buf.copyTo(out(cv::Rect(t.x, t.y, t.w, t.h)));
    return stats.elapsed_ms;
}
#endif

// ---- Main scheduler ----

TileSchedulerStats render_map_hybrid(
    const MapParams& p, cv::Mat& out,
    int tile_size, TileCallback on_tile_done
) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    auto tiles = make_tiles(p.width, p.height, tile_size);
    const size_t n = tiles.size();

    std::atomic<size_t> next_tile{0};

    const bool fx       = (p.scalar_type == "fx64") || (p.scalar_type == "auto" && p.scale < 1e-13);
    const bool use_avx  = avx512_available() && p.variant == Variant::Mandelbrot && p.metric == Metric::Escape;
    const bool use_gpu  = false
#if USE_CUDA
                       || (fsd_cuda::cuda_available() && p.variant == Variant::Mandelbrot && p.metric == Metric::Escape)
#endif
                       ;

    TileSchedulerStats result;
    std::mutex stats_mutex;
    double total_cpu_ms = 0.0, total_gpu_ms = 0.0;
    int cpu_tiles = 0, gpu_tiles = 0;

    const auto t_start = std::chrono::steady_clock::now();

    // GPU worker thread (if available)
    std::thread gpu_thread;
    WorkerStats gpu_ema;
    if (use_gpu) {
        gpu_thread = std::thread([&]() {
            while (true) {
                const size_t idx = next_tile.fetch_add(1);
                if (idx >= n) break;
                const Tile& tile = tiles[idx];
                const auto t0 = std::chrono::steady_clock::now();
#if USE_CUDA
                const double ms = render_tile_gpu(p, tile, out, fx);
#else
                (void)tile;
                const double ms = 0.0;
#endif
                const auto t1 = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double,std::milli>(t1-t0).count();
                gpu_ema.record(tile.w * tile.h, elapsed);
                {
                    std::lock_guard<std::mutex> lk(stats_mutex);
                    total_gpu_ms += ms;
                    gpu_tiles++;
                }
                if (on_tile_done) on_tile_done(tile.x, tile.y, tile.w, tile.h);
            }
        });
    }

    // CPU workers — one thread per available CPU core
    const int n_cpu_threads = default_render_threads();
    std::vector<std::thread> cpu_threads;
    cpu_threads.reserve(static_cast<size_t>(n_cpu_threads));
    WorkerStats cpu_ema;
    std::mutex cpu_ema_mutex;

    for (int tid = 0; tid < n_cpu_threads; tid++) {
        cpu_threads.emplace_back([&]() {
            while (true) {
                const size_t idx = next_tile.fetch_add(1);
                if (idx >= n) break;
                const Tile& tile = tiles[idx];
                const auto t0 = std::chrono::steady_clock::now();
                const double ms = render_tile_cpu(p, tile, out, use_avx, fx);
                const auto t1 = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double,std::milli>(t1-t0).count();
                {
                    std::lock_guard<std::mutex> lk(cpu_ema_mutex);
                    cpu_ema.record(tile.w * tile.h, elapsed);
                }
                {
                    std::lock_guard<std::mutex> lk(stats_mutex);
                    total_cpu_ms += ms;
                    cpu_tiles++;
                }
                if (on_tile_done) on_tile_done(tile.x, tile.y, tile.w, tile.h);
            }
        });
    }

    // Wait for all workers
    for (auto& t : cpu_threads) t.join();
    if (gpu_thread.joinable()) gpu_thread.join();

    const auto t_end = std::chrono::steady_clock::now();

    result.total_ms   = std::chrono::duration<double,std::milli>(t_end - t_start).count();
    result.cpu_ms     = total_cpu_ms;
    result.gpu_ms     = total_gpu_ms;
    result.cpu_tiles  = cpu_tiles;
    result.gpu_tiles  = gpu_tiles;
    result.scalar_used = fx ? "fx64" : "fp64";

    if (use_gpu && gpu_tiles > 0 && cpu_tiles > 0)
        result.engine_used = "hybrid";
    else if (use_gpu && gpu_tiles > 0)
        result.engine_used = "cuda";
    else if (use_avx)
        result.engine_used = "avx512";
    else
        result.engine_used = "openmp";

    return result;
}

} // namespace fsd::compute
