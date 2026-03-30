#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacyStlExport(const fs::path& repoRoot, const std::string& runDir) {
    const fs::path src = repoRoot / "cfiles" / "mandelbrot_3D.c";
    const fs::path exe = fs::path(runDir) / "stl_export.out";
    const fs::path stl = fs::path(runDir) / "mandelbrot_256.stl";
    const fs::path index = fs::path(runDir) / "mesh_index.json";

    const std::string compileCmd = "gcc \"" + src.string() + "\" -o \"" + exe.string() + "\" -lm";
    if (std::system(compileCmd.c_str()) != 0) {
        throw std::runtime_error("stl-export compile failed");
    }

    const std::string runCmd = "bash -lc 'cd \"" + runDir + "\" && \"" + exe.string() + "\"'";
    if (std::system(runCmd.c_str()) != 0) {
        throw std::runtime_error("stl-export execution failed");
    }

    std::ofstream out(index);
    out << "{\n"
        << "  \"module\": \"stl-export\",\n"
        << "  \"stl\": \"" << stl.filename().string() << "\",\n"
        << "  \"source\": \"cfiles/mandelbrot_3D.c\"\n"
        << "}\n";

    return Artifact{"stl-export", index.string(), "mesh-index"};
}

} // namespace fsd
