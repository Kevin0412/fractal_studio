// routes_modules.cpp — system check (openmp / cuda presence).
//
// Previously also dispatched legacy module runs — that surface is deleted.
// All native compute is exposed via dedicated endpoints (routes_map.cpp,
// routes_points.cpp, routes_hs.cpp, routes_ln.cpp, routes_video.cpp).

#include "routes.hpp"
#include "routes_common.hpp"
#include "system_checks.hpp"

namespace fsd {

std::string systemCheckRoute() {
    Json j = {
        {"openmp", checkOpenMP()},
        {"cuda",   checkCudaRuntime()},
    };
    return j.dump();
}

} // namespace fsd
