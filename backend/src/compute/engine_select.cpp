// compute/engine_select.cpp

#include "engine_select.hpp"

#include "map_kernel_avx512.hpp"
#include "parallel.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/map_kernel.cuh"
#  define USE_CUDA 1
#else
#  define USE_CUDA 0
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <mutex>
#include <thread>

namespace fsd::compute {
namespace {

std::mutex g_bench_mu;
std::vector<BenchmarkEntry> g_bench_entries;

int physical_core_guess(int logical) {
    return std::max(1, logical / 2);
}

bool is_cuda_metric_supported(Metric metric) {
    return static_cast<int>(metric) >= 0 && static_cast<int>(metric) < 4;
}

bool is_vector_metric_supported(Metric metric) {
    return static_cast<int>(metric) >= 0 && static_cast<int>(metric) < 4;
}

bool is_small_interactive_map(const MapParams& p) {
    const long long pixels = static_cast<long long>(p.width) * static_cast<long long>(p.height);
    const long long work = pixels * static_cast<long long>(std::max(1, p.iterations));
    return pixels < 900000LL || work < 600000000LL;
}

double cached_speed(const std::string& engine, const std::string& scalar) {
    std::lock_guard<std::mutex> lk(g_bench_mu);
    double best = 0.0;
    for (const auto& e : g_bench_entries) {
        if (!e.available) continue;
        if (e.engine == engine && e.scalar == scalar) best = std::max(best, e.mpix_per_sec);
    }
    return best;
}

} // namespace

RuntimeCapabilities runtime_capabilities() {
    RuntimeCapabilities c;
#ifdef _OPENMP
    c.openmp_compiled = true;
    c.openmp_runtime = omp_get_max_threads() > 0;
#else
    c.openmp_compiled = false;
    c.openmp_runtime = true;
#endif

#if defined(HAS_AVX512_KERNEL)
    c.avx512_compiled = true;
#endif
    c.avx512_runtime = avx512_available();

#if USE_CUDA
    c.cuda_compiled = true;
    c.cuda_runtime = fsd_cuda::cuda_available();
#endif

    c.logical_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    c.physical_cores = physical_core_guess(c.logical_cores);
    return c;
}

void update_benchmark_cache(const std::vector<BenchmarkEntry>& entries) {
    std::lock_guard<std::mutex> lk(g_bench_mu);
    g_bench_entries = entries;
}

bool has_benchmark_cache() {
    std::lock_guard<std::mutex> lk(g_bench_mu);
    return !g_bench_entries.empty();
}

std::vector<BenchmarkEntry> benchmark_cache_snapshot() {
    std::lock_guard<std::mutex> lk(g_bench_mu);
    return g_bench_entries;
}

bool map_engine_supported(const MapParams& p, const std::string& engine, bool fx) {
    if (p.variant == Variant::Custom || p.custom_step_fn) return engine == "openmp";
    if (p.smooth) return engine == "openmp";
    if (variant_needs_scalar_fallback(p.variant)) return engine == "openmp";
    if (fx && p.metric == Metric::MinPairwiseDist) return engine == "openmp";

    const RuntimeCapabilities c = runtime_capabilities();
    if (engine == "openmp") return true;

    if (engine == "avx512") {
        if (!c.avx512_runtime || !is_vector_metric_supported(p.metric)) return false;
        return !fx;
    }

    if (engine == "cuda") {
        return c.cuda_runtime && is_cuda_metric_supported(p.metric);
    }

    if (engine == "hybrid") {
        return c.cuda_runtime
            && is_cuda_metric_supported(p.metric)
            && map_work_is_large(p);
    }

    return false;
}

bool map_work_is_large(const MapParams& p) {
    return !is_small_interactive_map(p);
}

std::string select_map_engine(const MapParams& p, bool fx, const std::string& purpose) {
    if (p.engine != "auto") {
        if (p.engine == "hybrid" && !map_work_is_large(p)) {
            return map_engine_supported(p, "avx512", fx) ? "avx512" : "openmp";
        }
        return map_engine_supported(p, p.engine, fx) ? p.engine : "openmp";
    }

    if (p.variant == Variant::Custom || p.custom_step_fn || p.smooth || variant_needs_scalar_fallback(p.variant)) {
        return "openmp";
    }

    const std::string scalar = fx ? "fx64" : "fp64";
    const bool large = map_work_is_large(p);

    if (has_benchmark_cache()) {
        std::vector<std::string> candidates = {"openmp", "avx512", "cuda"};
        if (large || purpose == "batch" || purpose == "volume") candidates.push_back("hybrid");

        std::string best = "openmp";
        double best_speed = cached_speed("openmp", scalar);
        for (const auto& engine : candidates) {
            if (!map_engine_supported(p, engine, fx)) continue;
            const double speed = cached_speed(engine, scalar);
            if (speed > best_speed) {
                best_speed = speed;
                best = engine;
            }
        }
        return best;
    }

    if (!large) {
        if (map_engine_supported(p, "avx512", fx) && p.iterations >= 512) return "avx512";
        return "openmp";
    }

    if ((purpose == "batch" || purpose == "volume") && map_engine_supported(p, "hybrid", fx)) {
        return "hybrid";
    }
    if (map_engine_supported(p, "cuda", fx)) return "cuda";
    if (map_engine_supported(p, "avx512", fx)) return "avx512";
    return "openmp";
}

} // namespace fsd::compute
