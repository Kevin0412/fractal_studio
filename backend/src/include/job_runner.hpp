#pragma once

#include "types.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace fsd {

class JobRunner {
public:
    explicit JobRunner(std::filesystem::path runtimeRoot);

    RunRecord createRun(const std::string& module);
    RunRecord getRun(const std::string& runId) const;
    void setStatus(const std::string& runId, const std::string& status);
    void addArtifact(const std::string& runId, const Artifact& artifact);

private:
    std::filesystem::path runtimeRoot_;
    std::unordered_map<std::string, RunRecord> runs_;

    static std::string makeRunId();
};

} // namespace fsd
