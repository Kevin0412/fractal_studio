// routes_runs.cpp — run list from the persistent runs table.

#include "routes.hpp"
#include "routes_common.hpp"

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

} // namespace fsd
