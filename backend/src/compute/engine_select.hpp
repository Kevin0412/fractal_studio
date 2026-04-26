// compute/engine_select.hpp
//
// Runtime capability and benchmark-aware engine selection.

#pragma once

#include "map_kernel.hpp"

#include <string>
#include <vector>

namespace fsd::compute {

struct RuntimeCapabilities {
    bool openmp_compiled = false;
    bool openmp_runtime = false;
    bool avx512_compiled = false;
    bool avx512_runtime = false;
    bool cuda_compiled = false;
    bool cuda_runtime = false;
    int logical_cores = 1;
    int physical_cores = 1;
};

struct BenchmarkEntry {
    std::string engine;
    std::string scalar;
    double mpix_per_sec = 0.0;
    bool available = false;
};

RuntimeCapabilities runtime_capabilities();

void update_benchmark_cache(const std::vector<BenchmarkEntry>& entries);
bool has_benchmark_cache();
std::vector<BenchmarkEntry> benchmark_cache_snapshot();

bool map_engine_supported(const MapParams& p, const std::string& engine, bool fx);
bool map_work_is_large(const MapParams& p);
std::string select_map_engine(const MapParams& p, bool fx, const std::string& purpose = "map");

} // namespace fsd::compute
