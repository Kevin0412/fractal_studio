#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacyTransition(const fs::path& repoRoot, const std::string& runDir) {
    const fs::path src = repoRoot / "cfiles" / "mandelbrot_3Dtranslation_minmax.c";
    const fs::path exe = fs::path(runDir) / "transition.out";
    const fs::path png = fs::path(runDir) / "mandelbrot_3Dtranslation_minmax_256.png";
    const fs::path meta = fs::path(runDir) / "transition_manifest.json";

    const std::string compileCmd = "gcc \"" + src.string() + "\" -o \"" + exe.string() + "\" -lm";
    if (std::system(compileCmd.c_str()) != 0) {
        throw std::runtime_error("transition compile failed");
    }

    const std::string runCmd = "bash -lc 'cd \"" + runDir + "\" && \"" + exe.string() + "\" 30'";
    if (std::system(runCmd.c_str()) != 0) {
        throw std::runtime_error("transition execution failed");
    }

    std::ofstream out(meta);
    out << "{\n"
        << "  \"module\": \"transition-conversion\",\n"
        << "  \"theta_degrees\": 30,\n"
        << "  \"frame\": \"" << png.filename().string() << "\"\n"
        << "}\n";

    return Artifact{"transition-conversion", meta.string(), "dataset"};
}

} // namespace fsd
