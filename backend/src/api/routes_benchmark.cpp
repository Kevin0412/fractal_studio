// routes_benchmark.cpp
//
// POST /api/benchmark — runs a reference viewport through all engine×scalar
// combinations and returns a throughput table.
//
// Reference viewport: center=(-0.75, 0), scale=1.5, 1024×1024, iter=10000.
// Each engine runs once (no warmup for simplicity). Results are in Mpix/s.

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/map_kernel.hpp"
#include "../compute/map_kernel_avx512.hpp"
#include "../compute/engine_select.hpp"
#include "../compute/tile_scheduler.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "../compute/cuda/map_kernel.cuh"
#  define USE_CUDA 1
#else
#  define USE_CUDA 0
#endif

#include <opencv2/core.hpp>
#include <chrono>
#include <vector>

namespace fsd {

std::string benchmarkRoute(const std::string& body) {
    const Json jbody = body.empty() ? Json::object() : parseJsonBody(body);

    const double cRe   = jbody.value("centerRe",   -0.75);
    const double cIm   = jbody.value("centerIm",    0.0);
    const double scale = jbody.value("scale",       1.5);
    const int W        = jbody.value("width",       512);
    const int H        = jbody.value("height",      512);
    const int iters    = jbody.value("iterations",  2000);

    struct BenchResult {
        std::string engine;
        std::string scalar;
        double elapsed_ms;
        double mpix_per_sec;
        bool  available;
    };

    std::vector<BenchResult> results;

    auto run_bench = [&](const std::string& engine, const std::string& scalar) -> BenchResult {
        BenchResult r;
        r.engine   = engine;
        r.scalar   = scalar;
        r.available = false;
        r.elapsed_ms = 0.0;
        r.mpix_per_sec = 0.0;

        compute::MapParams p;
        p.center_re   = cRe;
        p.center_im   = cIm;
        p.scale       = scale;
        p.width       = W;
        p.height      = H;
        p.iterations  = iters;
        p.engine      = engine;
        p.scalar_type = scalar;

        cv::Mat out;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            if (engine == "hybrid") {
                auto stats = compute::render_map_hybrid(p, out);
                (void)stats;
            } else {
                auto stats = compute::render_map(p, out);
                (void)stats;
            }
            const auto t1 = std::chrono::steady_clock::now();
            r.elapsed_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
            r.mpix_per_sec = (static_cast<double>(W) * H / 1e6) / (r.elapsed_ms / 1000.0);
            r.available = true;
        } catch (...) {
            r.available = false;
        }
        return r;
    };

    // OpenMP fp64 / fx64
    results.push_back(run_bench("openmp", "fp64"));
    results.push_back(run_bench("openmp", "fx64"));

    // AVX-512 (only if available)
    if (compute::avx512_available()) {
        results.push_back(run_bench("avx512", "fp64"));
        results.push_back(run_bench("avx512", "fx64"));
    }

    // CUDA (only if available)
#if USE_CUDA
    if (fsd_cuda::cuda_available()) {
        // CUDA path via render_map — uses CUDA internally when engine="cuda"
        results.push_back(run_bench("cuda", "fp64"));
        results.push_back(run_bench("cuda", "fx64"));
        results.push_back(run_bench("hybrid", "fp64"));
        results.push_back(run_bench("hybrid", "fx64"));
    }
#endif

    Json jresults = Json::array();
    std::vector<compute::BenchmarkEntry> cache_entries;
    for (const auto& r : results) {
        jresults.push_back({
            {"engine",     r.engine},
            {"scalar",     r.scalar},
            {"available",  r.available},
            {"elapsedMs",  r.elapsed_ms},
            {"mpixPerSec", r.mpix_per_sec},
        });
        cache_entries.push_back({r.engine, r.scalar, r.mpix_per_sec, r.available});
    }
    compute::update_benchmark_cache(cache_entries);

    Json resp = {
        {"width",      W},
        {"height",     H},
        {"iterations", iters},
        {"results",    jresults},
    };
    return resp.dump();
}

} // namespace fsd
