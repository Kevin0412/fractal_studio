#include "path_guard.hpp"

namespace fs = std::filesystem;

namespace fsd {

PathGuard::PathGuard(fs::path repoRoot) : root_(fs::weakly_canonical(repoRoot)) {
    deniedRoots_ = {
        root_ / "C_mandelbrot",
        root_ / "cfiles",
        root_ / "cuda_mandelbrot",
        root_ / "Mandelbrot_set"
    };
}

bool PathGuard::isWriteAllowed(const fs::path& target) const {
    const auto canon = fs::weakly_canonical(target);
    for (const auto& denied : deniedRoots_) {
        if (canon.string().rfind(denied.string(), 0) == 0) {
            return false;
        }
    }
    return true;
}

std::string PathGuard::denyReason(const fs::path& target) const {
    if (isWriteAllowed(target)) {
        return "";
    }
    return "Write denied by legacy-immutability guard: " + target.string();
}

} // namespace fsd
