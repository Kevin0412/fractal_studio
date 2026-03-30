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

class Db {
public:
    explicit Db(std::filesystem::path dbPath);

    void ensureSchema() const;
    void insertSpecialPoint(const SpecialPointRecord& record) const;
    std::vector<SpecialPointRecord> listSpecialPoints(const std::string& familyFilter, int kFilter, int pFilter) const;

private:
    std::filesystem::path dbPath_;
};

std::string nowIso8601();
std::string makeId();

} // namespace fsd
