#include "job_runner.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

JobRunner::JobRunner(fs::path runtimeRoot) : runtimeRoot_(std::move(runtimeRoot)) {
    fs::create_directories(runtimeRoot_ / "runs");
}

RunRecord JobRunner::createRun(const std::string& module) {
    RunRecord run;
    run.id = makeRunId();
    run.status = "queued";
    run.outputDir = (runtimeRoot_ / "runs" / run.id).string();
    fs::create_directories(run.outputDir);

    run.artifacts.push_back(Artifact{module, run.outputDir + "/manifest.json", "manifest"});
    runs_[run.id] = run;
    return run;
}

RunRecord JobRunner::getRun(const std::string& runId) const {
    const auto it = runs_.find(runId);
    if (it == runs_.end()) {
        throw std::runtime_error("run not found: " + runId);
    }
    return it->second;
}

void JobRunner::setStatus(const std::string& runId, const std::string& status) {
    auto it = runs_.find(runId);
    if (it == runs_.end()) {
        throw std::runtime_error("run not found: " + runId);
    }
    it->second.status = status;
}

void JobRunner::addArtifact(const std::string& runId, const Artifact& artifact) {
    auto it = runs_.find(runId);
    if (it == runs_.end()) {
        throw std::runtime_error("run not found: " + runId);
    }
    it->second.artifacts.push_back(artifact);
}

std::string JobRunner::makeRunId() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

} // namespace fsd
