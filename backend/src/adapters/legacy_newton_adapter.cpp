#include "adapters.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

Artifact runLegacySpecialPoints(const fs::path& repoRoot, const std::string& runDir) {
    const fs::path legacyDir = repoRoot / "C_mandelbrot";
    const fs::path scriptPath = fs::path(runDir) / "run_special_points.py";
    const fs::path jsonPath = fs::path(runDir) / "special_points.json";
    const fs::path csvPath = fs::path(runDir) / "special_points.csv";

    std::ofstream py(scriptPath);
    py << "import json\n"
       << "import csv\n"
       << "import sys\n"
       << "sys.path.insert(0, r'" << legacyDir.string() << "')\n"
       << "import special_points_of_mandelbrot_set as sp\n"
       << "points = sp.solution(1,1)\n"
       << "with open(r'" << jsonPath.string() << "', 'w', encoding='utf-8') as f:\n"
       << "    json.dump([{'real': float(p.real), 'imag': float(p.imag)} for p in points], f, ensure_ascii=False, indent=2)\n"
       << "with open(r'" << csvPath.string() << "', 'w', newline='', encoding='utf-8') as f:\n"
       << "    w = csv.writer(f)\n"
       << "    w.writerow(['real','imag'])\n"
       << "    for p in points:\n"
       << "        w.writerow([float(p.real), float(p.imag)])\n";

    const std::string cmd = "python3 \"" + scriptPath.string() + "\"";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("special-points execution failed");
    }

    return Artifact{"special-points", jsonPath.string(), "dataset"};
}

} // namespace fsd
