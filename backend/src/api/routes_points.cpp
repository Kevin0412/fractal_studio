#include "routes.hpp"

#include "db.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace fsd {

namespace {

std::optional<std::string> getStringField(const std::string& body, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(body, m, re)) {
        return m[1].str();
    }
    return std::nullopt;
}

std::optional<int> getIntField(const std::string& body, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (std::regex_search(body, m, re)) {
        return std::stoi(m[1].str());
    }
    return std::nullopt;
}

std::optional<double> getNumberField(const std::string& body, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch m;
    if (std::regex_search(body, m, re)) {
        return std::stod(m[1].str());
    }
    return std::nullopt;
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

std::vector<std::pair<double, double>> parsePointList(const std::string& json) {
    const std::regex pointRe("\\{\\s*\"real\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*,\\s*\"imag\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*\\}");
    std::vector<std::pair<double, double>> out;
    auto begin = std::sregex_iterator(json.begin(), json.end(), pointRe);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        out.emplace_back(std::stod((*it)[1].str()), std::stod((*it)[2].str()));
    }
    return out;
}

std::string readText(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string getQueryParam(const std::string& query, const std::string& key) {
    const std::regex re("(?:^|&)" + key + "=([^&]*)");
    std::smatch m;
    if (std::regex_search(query, m, re)) {
        return m[1].str();
    }
    return "";
}

Db openDb(const std::filesystem::path& repoRoot) {
    const fs::path dbDir = repoRoot / "fractal_studio" / "runtime" / "db";
    fs::create_directories(dbDir);
    Db db(dbDir / "fractal_studio.sqlite3");
    db.ensureSchema();
    return db;
}

} // namespace

std::string specialPointsAutoRoute(const std::filesystem::path& repoRoot, const std::string& body) {
    const int k = getIntField(body, "k").value_or(1);
    const int p = getIntField(body, "p").value_or(1);
    const std::string family = getStringField(body, "family").value_or("mandelbrot");
    const std::string pointType = getStringField(body, "pointType").value_or("misiurewicz");

    if (pointType == "misiurewicz" && k <= 0) {
        throw std::runtime_error("misiurewicz requires pre-period k > 0");
    }
    if (pointType == "hyperbolic" && k != 0) {
        throw std::runtime_error("hyperbolic requires k = 0");
    }

    const fs::path runtimeDir = repoRoot / "fractal_studio" / "runtime";
    fs::create_directories(runtimeDir);
    const std::string id = makeId();
    const fs::path scriptPath = runtimeDir / ("special_points_auto_" + id + ".py");
    const fs::path outputPath = runtimeDir / ("special_points_auto_" + id + ".json");

    const fs::path legacyDir = repoRoot / "C_mandelbrot";
    std::ofstream py(scriptPath);
    py << "import json\n"
       << "import sys\n"
       << "sys.path.insert(0, r'" << legacyDir.string() << "')\n"
       << "import special_points_of_mandelbrot_set as sp\n"
       << "points = sp.solution(" << k << "," << p << ")\n"
       << "with open(r'" << outputPath.string() << "', 'w', encoding='utf-8') as f:\n"
       << "    json.dump([{'real': float(x.real), 'imag': float(x.imag)} for x in points], f, ensure_ascii=False)\n";

    py.flush();
    py.close();

    const std::string cmd = "python3 \"" + scriptPath.string() + "\"";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("special-points auto execution failed");
    }

    const std::string raw = readText(outputPath);
    const auto points = parsePointList(raw);

    Db db = openDb(repoRoot);
    const std::string createdAt = nowIso8601();
    std::vector<std::string> insertedIds;
    insertedIds.reserve(points.size());

    for (const auto& point : points) {
        SpecialPointRecord record;
        record.id = makeId();
        record.family = family;
        record.pointType = pointType;
        record.k = k;
        record.p = p;
        record.re = point.first;
        record.im = point.second;
        record.sourceMode = "auto";
        record.createdAt = createdAt;
        db.insertSpecialPoint(record);
        insertedIds.push_back(record.id);
    }

    std::ostringstream ss;
    ss << "{\"mode\":\"auto\",\"k\":" << k << ",\"p\":" << p << ",\"count\":" << points.size() << ",\"points\":[";
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << "{\"id\":\"" << insertedIds[i] << "\",\"real\":" << points[i].first << ",\"imag\":" << points[i].second << "}";
    }
    ss << "]}";
    return ss.str();
}

std::string specialPointsSeedRoute(const std::filesystem::path& repoRoot, const std::string& body) {
    const int k = getIntField(body, "k").value_or(1);
    const int p = getIntField(body, "p").value_or(1);
    const int maxIter = getIntField(body, "maxIter").value_or(256);
    const double seedRe = getNumberField(body, "re").value_or(0.0);
    const double seedIm = getNumberField(body, "im").value_or(0.0);
    const std::string family = getStringField(body, "family").value_or("mandelbrot");

    const fs::path runtimeDir = repoRoot / "fractal_studio" / "runtime";
    fs::create_directories(runtimeDir);
    const std::string id = makeId();
    const fs::path scriptPath = runtimeDir / ("special_points_seed_" + id + ".py");
    const fs::path outputPath = runtimeDir / ("special_points_seed_" + id + ".json");

    const fs::path legacyDir = repoRoot / "C_mandelbrot";
    std::ofstream py(scriptPath);
    py << "import json\n"
       << "import sys\n"
       << "sys.path.insert(0, r'" << legacyDir.string() << "')\n"
       << "import newton_Mandelbrot as nm\n"
       << "seed = complex(" << seedRe << "," << seedIm << ")\n"
       << "sol = nm.solve(" << k << "," << p << ", seed, " << maxIter << ")\n"
       << "pt = sol[0]\n"
       << "res = sol[2]\n"
       << "with open(r'" << outputPath.string() << "', 'w', encoding='utf-8') as f:\n"
       << "    json.dump({'real': float(pt.real), 'imag': float(pt.imag), 'digits': int(sol[1]), 'residual_real': float(res.real), 'residual_imag': float(res.imag)}, f, ensure_ascii=False)\n";

    py.flush();
    py.close();

    const std::string cmd = "python3 \"" + scriptPath.string() + "\"";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("special-points seed execution failed");
    }

    const std::string raw = readText(outputPath);
    const double re = getNumberField(raw, "real").value_or(0.0);
    const double im = getNumberField(raw, "imag").value_or(0.0);
    const int digits = getIntField(raw, "digits").value_or(0);
    const double residualRe = getNumberField(raw, "residual_real").value_or(0.0);
    const double residualIm = getNumberField(raw, "residual_imag").value_or(0.0);

    Db db = openDb(repoRoot);
    SpecialPointRecord record;
    record.id = makeId();
    record.family = family;
    record.pointType = "newton";
    record.k = k;
    record.p = p;
    record.re = re;
    record.im = im;
    record.sourceMode = "seed";
    record.createdAt = nowIso8601();
    db.insertSpecialPoint(record);

    std::ostringstream ss;
    ss << "{\"mode\":\"seed\",\"point\":{\"id\":\"" << record.id << "\",\"real\":" << re << ",\"imag\":" << im
       << ",\"digits\":" << digits << ",\"residualReal\":" << residualRe << ",\"residualImag\":" << residualIm << "}}";
    return ss.str();
}

std::string specialPointsListRoute(const std::filesystem::path& repoRoot, const std::string& query) {
    const std::string family = getQueryParam(query, "family");

    int k = -1;
    int p = -1;
    const std::string kRaw = getQueryParam(query, "k");
    const std::string pRaw = getQueryParam(query, "p");
    if (!kRaw.empty()) {
        k = std::stoi(kRaw);
    }
    if (!pRaw.empty()) {
        p = std::stoi(pRaw);
    }

    Db db = openDb(repoRoot);
    const auto rows = db.listSpecialPoints(family, k, p);

    std::ostringstream ss;
    ss << "{\"items\":[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        const auto& row = rows[i];
        ss << "{"
           << "\"id\":\"" << jsonEscape(row.id) << "\","
           << "\"family\":\"" << jsonEscape(row.family) << "\","
           << "\"pointType\":\"" << jsonEscape(row.pointType) << "\","
           << "\"k\":" << row.k << ","
           << "\"p\":" << row.p << ","
           << "\"real\":" << row.re << ","
           << "\"imag\":" << row.im << ","
           << "\"sourceMode\":\"" << jsonEscape(row.sourceMode) << "\","
           << "\"createdAt\":\"" << jsonEscape(row.createdAt) << "\""
           << "}";
    }
    ss << "]}";
    return ss.str();
}

} // namespace fsd
