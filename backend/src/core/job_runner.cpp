#include "job_runner.hpp"
#include "db.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

JobRunner::JobRunner(fs::path runtimeRoot, Db* db)
    : runtimeRoot_(std::move(runtimeRoot)), db_(db) {
    fs::create_directories(runtimeRoot_ / "runs");
}

RunRecord JobRunner::createRun(const std::string& module, const std::string& paramsJson) {
    RunRecord run;
    run.id = makeRunId();
    run.status = "queued";
    run.outputDir = (runtimeRoot_ / "runs" / run.id).string();
    fs::create_directories(run.outputDir);

    const long long started = nowUnixMs();
    {
        std::lock_guard<std::mutex> lk(mu_);
        runs_[run.id] = run;
        runModules_[run.id] = module;
        runParams_[run.id] = paramsJson;
        runStarted_[run.id] = started;
    }

    if (db_) {
        RunRow row{run.id, module, run.status, paramsJson, started, 0, run.outputDir};
        std::lock_guard<std::mutex> lk(mu_);
        db_->upsertRun(row);
    }
    return run;
}

RunRecord JobRunner::getRun(const std::string& runId) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = runs_.find(runId);
    if (it == runs_.end()) {
        throw std::runtime_error("run not found: " + runId);
    }
    return it->second;
}

void JobRunner::setStatus(const std::string& runId, const std::string& status) {
    std::string module, paramsJson, outDir;
    long long started = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = runs_.find(runId);
        if (it == runs_.end()) throw std::runtime_error("run not found: " + runId);
        it->second.status = status;
        outDir = it->second.outputDir;
        module = runModules_[runId];
        paramsJson = runParams_[runId];
        started = runStarted_[runId];
    }
    if (db_) {
        const long long finished = (status == "completed" || status == "failed") ? nowUnixMs() : 0;
        RunRow row{runId, module, status, paramsJson, started, finished, outDir};
        std::lock_guard<std::mutex> lk(mu_);
        db_->upsertRun(row);
    }
}

void JobRunner::setProgress(const std::string& runId, const std::string& progressJson) {
    std::string outDir;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = runs_.find(runId);
        if (it == runs_.end()) throw std::runtime_error("run not found: " + runId);
        runProgress_[runId] = progressJson;
        outDir = it->second.outputDir;
    }
    if (!outDir.empty()) {
        std::ofstream os(fs::path(outDir) / "progress.json");
        os << progressJson;
    }
}

std::string JobRunner::getProgress(const std::string& runId) const {
    std::string outDir;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = runProgress_.find(runId);
        if (it != runProgress_.end()) return it->second;
        auto runIt = runs_.find(runId);
        if (runIt != runs_.end()) outDir = runIt->second.outputDir;
    }
    if (outDir.empty()) {
        outDir = resolveOutputDir(runId);
    }
    if (!outDir.empty()) {
        std::ifstream is(fs::path(outDir) / "progress.json");
        if (is) {
            std::string text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
            if (!text.empty()) return text;
        }
    }
    return "{}";
}

void JobRunner::addArtifact(const std::string& runId, const Artifact& artifact) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = runs_.find(runId);
        if (it == runs_.end()) throw std::runtime_error("run not found: " + runId);
        it->second.artifacts.push_back(artifact);
    }
    if (db_) {
        ArtifactRow row{0, runId, artifact.kind, artifact.path, ""};
        std::lock_guard<std::mutex> lk(mu_);
        db_->insertArtifact(row);
    }
}

std::string JobRunner::resolveOutputDir(const std::string& runId) const {
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = runs_.find(runId);
        if (it != runs_.end()) return it->second.outputDir;
    }
    if (db_) {
        try {
            return db_->getRun(runId).outputDir;
        } catch (...) {}
    }
    return {};
}

std::string JobRunner::makeRunId() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

} // namespace fsd
