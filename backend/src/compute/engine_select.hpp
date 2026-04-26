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
    bool avx2_compiled = false;
    bool avx2_runtime = false;
    bool fma_runtime = false;
    bool bmi2_runtime = false;
    bool avx512_compiled = false;
    bool avx512_runtime = false;
    bool avx512ifma_runtime = false;
    bool cuda_compiled = false;
    bool cuda_runtime = false;
    bool cuda_low_end = false;
    int cuda_device_count = 0;
    int cuda_compute_major = 0;
    int cuda_compute_minor = 0;
    unsigned long long cuda_total_vram = 0;
    unsigned long long cuda_free_vram = 0;
    std::string cuda_name;
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
