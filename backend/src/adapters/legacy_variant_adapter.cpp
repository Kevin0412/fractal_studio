#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacyVariantAtlas(const fs::path& repoRoot, const std::string& runDir) {
    const fs::path legacyDir = repoRoot / "C_mandelbrot";
    const fs::path buildExe = fs::path(runDir) / "Mandelbrot_python_ln.out";
    const fs::path imagePath = fs::path(runDir) / "atlas_variety0.png";
    const fs::path atlasManifest = fs::path(runDir) / "atlas_manifest.json";

    const std::string compileCmd = "gcc \"" + (legacyDir / "Mandelbrot_python_ln.c").string() +
                                   "\" -o \"" + buildExe.string() + "\" -lm";
    if (std::system(compileCmd.c_str()) != 0) {
        throw std::runtime_error("atlas compile failed");
    }

    const std::string runCmd = "\"" + buildExe.string() + "\" 0 0 0 \"" + imagePath.string() + "\" 0 0 0";
    if (std::system(runCmd.c_str()) != 0) {
        throw std::runtime_error("atlas execution failed");
    }

    std::ofstream out(atlasManifest);
    out << "{\n"
        << "  \"module\": \"atlas\",\n"
        << "  \"legacy_source\": \"C_mandelbrot/Mandelbrot_python_ln.c\",\n"
        << "  \"image\": \"" << imagePath.filename().string() << "\",\n"
        << "  \"mode\": \"ln_exp\",\n"
        << "  \"variety\": 0\n"
        << "}\n";

    return Artifact{"atlas", atlasManifest.string(), "dataset"};
}

} // namespace fsd
