#include "system_checks.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace fsd {

bool checkOpenMP() {
#ifdef _OPENMP
    return omp_get_max_threads() > 0;
#else
    return false;
#endif
}

} // namespace fsd
