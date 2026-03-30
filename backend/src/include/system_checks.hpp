#pragma once

namespace fsd {

struct SystemCheckResult {
    bool openmpOk;
    bool cudaOk;
};

bool checkOpenMP();
bool checkCudaRuntime();

} // namespace fsd
