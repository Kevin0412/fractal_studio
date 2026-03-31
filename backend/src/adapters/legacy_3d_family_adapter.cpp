#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacyHiddenStructureFamily(const fs::path& repoRoot, const std::string& runDir) {
    struct StageCmd {
        std::string stage;
        fs::path src;
    };

    const std::vector<StageCmd> stages = {
        {"HS-Base", repoRoot / "cfiles" / "mandelbrot_3D.c"},
        {"HS-Variant", repoRoot / "cfiles" / "mandelbrot_3D2.c"},
        {"HS-Recurrence", repoRoot / "cfiles" / "mandelbrot_3D3.c"},
        {"HS-Symmetry", repoRoot / "cfiles" / "duck_3D4.c"},
        {"HS-Envelope", repoRoot / "cfiles" / "mandelbrot_3D5_maxmin.c"},
        {"HS-MultiChannel", repoRoot / "cfiles" / "mandelbrot_3D6_2D_enlarge.c"}
    };

    const fs::path matrix = fs::path(runDir) / "hs_family_matrix.json";
    std::ofstream out(matrix);
    out << "{\n  \"module\": \"hidden-structure-family\",\n  \"stages\": [\n";

    const unsigned int hw = std::thread::hardware_concurrency();
    const unsigned int ompThreads = hw > 1 ? hw - 1 : 1;

    for (size_t i = 0; i < stages.size(); ++i) {
        const auto& s = stages[i];
        const fs::path exe = fs::path(runDir) / (s.stage + ".out");

        std::string compileCmd = "gcc \"" + s.src.string() + "\" -o \"" + exe.string() + "\" -lm -std=gnu11 -include stdint.h -O3";
        if (s.stage == "HS-MultiChannel" || s.stage == "HS-Symmetry") {
            compileCmd += " -fopenmp";
        }

        int rc = std::system(compileCmd.c_str());
        bool runOk = false;
        if (rc == 0) {
            std::string runCmd = "bash -lc 'cd \"" + runDir + "\" && \"" + exe.string() + "\"'";
            if (s.stage == "HS-MultiChannel" || s.stage == "HS-Symmetry") {
                runCmd = "bash -lc 'cd \"" + runDir + "\" && OMP_NUM_THREADS=" + std::to_string(ompThreads) + " \"" + exe.string() + "\"'";
            }
            runOk = (std::system(runCmd.c_str()) == 0);
        }

        out << "    {\"stage\":\"" << s.stage << "\",\"source\":\"" << s.src.filename().string()
            << "\",\"compile_ok\":" << (rc == 0 ? "true" : "false")
            << ",\"run_ok\":" << (runOk ? "true" : "false")
            << ",\"omp_threads\":" << ((s.stage == "HS-MultiChannel" || s.stage == "HS-Symmetry") ? std::to_string(ompThreads) : "0")
            << "}";
        if (i + 1 != stages.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n}\n";
    return Artifact{"hidden-structure-family", matrix.string(), "dataset"};
}

} // namespace fsd
