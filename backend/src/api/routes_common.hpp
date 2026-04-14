// routes_common.hpp
//
// Shared JSON/query helpers and Db opener used across the API route files.
// Headers-only so every route TU gets the same helpers.

#pragma once

#include "db.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cstdlib>
#include <filesystem>
#include <regex>
#include <string>

namespace fsd {

using Json = nlohmann::json;

inline Json parseJsonBody(const std::string& body) {
    if (body.empty()) return Json::object();
    try {
        return Json::parse(body);
    } catch (const std::exception&) {
        return Json::object();
    }
}

inline std::string getQueryParam(const std::string& query, const std::string& key) {
    const std::regex re("(?:^|&)" + key + "=([^&]*)");
    std::smatch m;
    if (std::regex_search(query, m, re)) {
        return m[1].str();
    }
    return "";
}

inline std::string urlDecode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const std::string hex = s.substr(i + 1, 2);
            const char c = static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
            out.push_back(c);
            i += 2;
        } else if (s[i] == '+') {
            out.push_back(' ');
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

inline Db openDb(const std::filesystem::path& repoRoot) {
    namespace fs = std::filesystem;
    const fs::path dbDir = repoRoot / "fractal_studio" / "runtime" / "db";
    fs::create_directories(dbDir);
    Db db(dbDir / "fractal_studio.sqlite3");
    db.ensureSchema();
    return db;
}

inline std::filesystem::path repoRuntime(const std::filesystem::path& repoRoot) {
    return repoRoot / "fractal_studio" / "runtime";
}

} // namespace fsd
