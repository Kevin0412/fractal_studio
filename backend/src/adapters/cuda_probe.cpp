#include "system_checks.hpp"

#include <cstdlib>

namespace fsd {

bool checkCudaRuntime() {
    const int rc = std::system("nvcc --version > /dev/null 2>&1");
    return rc == 0;
}

} // namespace fsd
