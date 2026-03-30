#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacyHiddenStructure(const fs::path& repoRoot, const std::string& runDir) {
    const fs::path src = repoRoot / "cfiles" / "mandelbrot_3D3_2D_256_2.c";
    const fs::path exe = fs::path(runDir) / "hidden_structure.out";
    const fs::path png = fs::path(runDir) / "mandelbrot_3D3_2D_256_2.png";
    const fs::path json = fs::path(runDir) / "hidden_structure.json";

    const std::string compileCmd = "gcc \"" + src.string() + "\" -o \"" + exe.string() + "\" -lm";
    if (std::system(compileCmd.c_str()) != 0) {
        throw std::runtime_error("hidden-structure compile failed");
    }

    const std::string runCmd = "bash -lc 'cd \"" + runDir + "\" && \"" + exe.string() + "\"'";
    if (std::system(runCmd.c_str()) != 0) {
        throw std::runtime_error("hidden-structure execution failed");
    }

    std::ofstream out(json);
    out << "{\n"
        << "  \"module\": \"hidden-structure\",\n"
        << "  \"metric\": \"min(|a_n-a_m|)\",\n"
        << "  \"stage\": \"HS-Recurrence\",\n"
        << "  \"image\": \"" << png.filename().string() << "\"\n"
        << "}\n";

    return Artifact{"hidden-structure", json.string(), "dataset"};
}

} // namespace fsd
