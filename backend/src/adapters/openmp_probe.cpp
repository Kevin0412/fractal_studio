#include "system_checks.hpp"

#include <omp.h>

namespace fsd {

bool checkOpenMP() {
    return omp_get_max_threads() > 0;
}

} // namespace fsd
