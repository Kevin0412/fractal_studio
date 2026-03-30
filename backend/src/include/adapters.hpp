#pragma once

#include "types.hpp"

#include <filesystem>
#include <string>

namespace fsd {

Artifact runLegacyVariantAtlas(const std::filesystem::path& repoRoot, const std::string& runDir);
Artifact runLegacySpecialPoints(const std::filesystem::path& repoRoot, const std::string& runDir);
Artifact runLegacyTransition(const std::filesystem::path& repoRoot, const std::string& runDir);
Artifact runLegacyHiddenStructure(const std::filesystem::path& repoRoot, const std::string& runDir);
Artifact runLegacyHiddenStructureFamily(const std::filesystem::path& repoRoot, const std::string& runDir);
Artifact runLegacyStlExport(const std::filesystem::path& repoRoot, const std::string& runDir);

struct MapRenderParams {
    double centerRe;
    double centerIm;
    double scale;
    int width;
    int height;
    int variety;
    int iterations;
    std::string colorMap;
};

Artifact runManagedMapRender(const std::filesystem::path& repoRoot, const std::string& runDir, const MapRenderParams& params);

} // namespace fsd
