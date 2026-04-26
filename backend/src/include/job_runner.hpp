#pragma once

#include "types.hpp"

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace fsd {

class Db;

class JobRunner {
public:
    // Takes an optional persistent Db pointer. When set, run state is written
    // through to sqlite on every transition so runs survive a restart.
    explicit JobRunner(std::filesystem::path runtimeRoot, Db* db = nullptr);

    RunRecord createRun(const std::string& module, const std::string& paramsJson = "");
    RunRecord getRun(const std::string& runId) const;
    void setStatus(const std::string& runId, const std::string& status);
    void setProgress(const std::string& runId, const std::string& progressJson);
    std::string getProgress(const std::string& runId) const;
    void addArtifact(const std::string& runId, const Artifact& artifact);

    // Lookup the on-disk path for a run by id, consulting sqlite on miss.
    std::string resolveOutputDir(const std::string& runId) const;

private:
    std::filesystem::path runtimeRoot_;
    Db* db_ = nullptr;
    mutable std::mutex mu_;
    std::unordered_map<std::string, RunRecord> runs_;
    std::unordered_map<std::string, std::string> runModules_;
    std::unordered_map<std::string, std::string> runParams_;
    std::unordered_map<std::string, std::string> runProgress_;
    std::unordered_map<std::string, long long>   runStarted_;

    static std::string makeRunId();
};

} // namespace fsd
