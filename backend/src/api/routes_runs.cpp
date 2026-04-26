// routes_runs.cpp — run list from the persistent runs table.

#include "routes.hpp"
#include "routes_common.hpp"

#include <filesystem>

namespace fsd {

std::string runsListRoute(const std::filesystem::path& repoRoot, const std::string& query) {
    int limit = 200;
    const std::string limRaw = getQueryParam(query, "limit");
    if (!limRaw.empty()) {
        try { limit = std::stoi(limRaw); } catch (...) {}
    }
    Db db = openDb(repoRoot);
    const auto rows = db.listRuns(limit);

    Json items = Json::array();
    for (const auto& r : rows) {
        items.push_back({
            {"id",         r.id},
            {"module",     r.module},
            {"status",     r.status},
            {"startedAt",  r.startedAt},
            {"finishedAt", r.finishedAt},
            {"outputDir",  r.outputDir},
        });
    }
    Json resp = {{"items", items}};
    return resp.dump();
}

std::string runStatusRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& query) {
    const std::string runId = getQueryParam(query, "runId");
    if (runId.empty()) throw std::runtime_error("runId required");

    Db db = openDb(repoRoot);
    RunRow row = db.getRun(runId);
    Json progress = Json::object();
    try {
        const std::string progressText = runner.getProgress(runId);
        if (!progressText.empty()) progress = Json::parse(progressText);
    } catch (...) {
        progress = Json::object();
    }

    Json artifacts = Json::array();
    for (const auto& a : db.listArtifacts(runId)) {
        const std::filesystem::path p(a.path);
        const std::string fileName = p.filename().string();
        const std::string artifactId = runId + ":" + fileName;
        artifacts.push_back({
            {"artifactId", artifactId},
            {"name", fileName},
            {"kind", a.kind},
            {"downloadUrl", "/api/artifacts/download?artifactId=" + artifactId},
            {"contentUrl", "/api/artifacts/content?artifactId=" + artifactId},
        });
    }

    Json resp = {
        {"id", row.id},
        {"module", row.module},
        {"status", row.status},
        {"startedAt", row.startedAt},
        {"finishedAt", row.finishedAt},
        {"outputDir", row.outputDir},
        {"progress", progress},
        {"artifacts", artifacts},
    };
    return resp.dump();
}

} // namespace fsd
