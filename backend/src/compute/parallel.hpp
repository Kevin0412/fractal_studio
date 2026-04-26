// compute/parallel.hpp
//
// Shared CPU parallelism policy for render hot paths.

#pragma once

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace fsd::compute {

inline int env_render_threads() noexcept {
    const char* raw = std::getenv("FSD_RENDER_THREADS");
    if (!raw || *raw == '\0') return 0;

    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed <= 0 || parsed > 4096) return 0;
    return static_cast<int>(parsed);
}

inline bool thermal_friendly_mode() noexcept {
    const char* raw = std::getenv("FSD_THERMAL_FRIENDLY");
    if (!raw || *raw == '\0') return false;
    return raw[0] == '1' || raw[0] == 't' || raw[0] == 'T' ||
           raw[0] == 'y' || raw[0] == 'Y';
}

inline int default_render_threads() noexcept {
    const int env_threads = env_render_threads();
    if (env_threads > 0) return env_threads;

    int threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 1;

#ifdef _OPENMP
    threads = std::max(threads, omp_get_num_procs());
#endif

    if (thermal_friendly_mode()) {
        threads = std::max(1, (threads + 1) / 2);
    }

    return std::max(1, threads);
}

inline int resolve_render_threads(int requested_threads) noexcept {
    return requested_threads > 0
        ? std::max(1, requested_threads)
        : default_render_threads();
}

} // namespace fsd::compute
