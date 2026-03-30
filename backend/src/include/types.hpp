#pragma once

#include <string>
#include <vector>

namespace fsd {

struct Artifact {
    std::string module;
    std::string path;
    std::string kind;
};

struct RunRecord {
    std::string id;
    std::string status;
    std::string outputDir;
    std::vector<Artifact> artifacts;
};

} // namespace fsd
