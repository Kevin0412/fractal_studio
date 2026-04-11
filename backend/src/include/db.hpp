#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fsd {

struct SpecialPointRecord {
    std::string id;
    std::string family;
    std::string pointType;
    int k;
    int p;
    double re;
    double im;
    std::string sourceMode;
    std::string createdAt;
};

struct RunRow {
    std::string id;
    std::string module;
    std::string status;
    std::string paramsJson;
    long long startedAt;   // unix ms
    long long finishedAt;  // unix ms, 0 if unfinished
    std::string outputDir;
};

struct ArtifactRow {
    long long rowId;
    std::string runId;
    std::string kind;
    std::string path;
    std::string metaJson;
};

class Db {
public:
    explicit Db(std::filesystem::path dbPath);

    void ensureSchema() const;

    // Special points
    void insertSpecialPoint(const SpecialPointRecord& record) const;
    std::vector<SpecialPointRecord> listSpecialPoints(const std::string& familyFilter, int kFilter, int pFilter) const;

    // Runs + artifacts
    void upsertRun(const RunRow& row) const;
    std::vector<RunRow> listRuns(int limit) const;
    RunRow getRun(const std::string& runId) const;

    long long insertArtifact(const ArtifactRow& row) const;
    std::vector<ArtifactRow> listArtifacts(const std::string& runId) const;
    std::vector<ArtifactRow> listArtifactsByKind(const std::string& kind, int limit) const;
    ArtifactRow getArtifactById(long long rowId) const;

private:
    std::filesystem::path dbPath_;
};

std::string nowIso8601();
long long nowUnixMs();
std::string makeId();

} // namespace fsd
