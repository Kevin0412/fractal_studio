// routes_modules.cpp — system check (openmp / cuda presence).
//
// Previously also dispatched legacy module runs — that surface is deleted.
// All native compute is exposed via dedicated endpoints (routes_map.cpp,
// routes_points.cpp, routes_hs.cpp, routes_ln.cpp, routes_video.cpp).

#include "routes.hpp"
#include "routes_common.hpp"
#include "system_checks.hpp"

#include "../compute/engine_select.hpp"

namespace fsd {

std::string systemCheckRoute() {
    Json j = {
        {"openmp", checkOpenMP()},
        {"cuda",   checkCudaRuntime()},
    };
    return j.dump();
}

std::string systemCapabilitiesRoute() {
    const auto caps = compute::runtime_capabilities();
    Json bench = Json::array();
    for (const auto& e : compute::benchmark_cache_snapshot()) {
        bench.push_back({
            {"engine", e.engine},
            {"scalar", e.scalar},
            {"available", e.available},
            {"mpixPerSec", e.mpix_per_sec},
        });
    }
    Json j = {
        {"openmp", {
            {"compiled", caps.openmp_compiled},
            {"runtime", caps.openmp_runtime},
        }},
        {"avx512", {
            {"compiled", caps.avx512_compiled},
            {"runtime", caps.avx512_runtime},
        }},
        {"cuda", {
            {"compiled", caps.cuda_compiled},
            {"runtime", caps.cuda_runtime},
        }},
        {"cpu", {
            {"logicalCores", caps.logical_cores},
            {"physicalCores", caps.physical_cores},
        }},
        {"benchmarkCache", {
            {"available", compute::has_benchmark_cache()},
            {"results", bench},
        }},
    };
    return j.dump();
}

} // namespace fsd
