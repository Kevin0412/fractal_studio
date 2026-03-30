#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fsd {

class PathGuard {
public:
    explicit PathGuard(std::filesystem::path repoRoot);

    bool isWriteAllowed(const std::filesystem::path& target) const;
    std::string denyReason(const std::filesystem::path& target) const;

private:
    std::filesystem::path root_;
    std::vector<std::filesystem::path> deniedRoots_;
};

} // namespace fsd
