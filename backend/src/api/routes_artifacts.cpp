#include "routes.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace fsd {

namespace {

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

std::string getQueryParam(const std::string& query, const std::string& key) {
    const std::regex re("(?:^|&)" + key + "=([^&]*)");
    std::smatch m;
    if (std::regex_search(query, m, re)) {
        return m[1].str();
    }
    return "";
}

std::string lowerExt(const fs::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return ext;
}

std::string kindFromPath(const fs::path& path) {
    const std::string ext = lowerExt(path);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".gif" || ext == ".webp") {
        return "image";
    }
    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".webm") {
        return "video";
    }
    if (ext == ".stl") {
        return "stl";
    }
    if (ext == ".json" || ext == ".csv" || ext == ".txt" || ext == ".md") {
        return "report";
    }
    return "other";
}

std::string contentTypeFromPath(const fs::path& path) {
    const std::string ext = lowerExt(path);
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif") return "image/gif";
    if (ext == ".webp") return "image/webp";
    if (ext == ".bmp") return "image/bmp";
    if (ext == ".mp4") return "video/mp4";
    if (ext == ".webm") return "video/webm";
    if (ext == ".mov") return "video/quicktime";
    if (ext == ".stl") return "application/sla";
    if (ext == ".json") return "application/json";
    if (ext == ".csv") return "text/csv";
    return "application/octet-stream";
}

std::string readFileText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct ArtifactEntry {
    std::string artifactId;
    std::string runId;
    std::string fileName;
    std::string path;
    std::string kind;
};

std::vector<ArtifactEntry> collectArtifacts(const fs::path& repoRoot) {
    const fs::path runsRoot = repoRoot / "fractal_studio" / "runtime" / "runs";
    std::vector<ArtifactEntry> rows;
    if (!fs::exists(runsRoot)) {
        return rows;
    }

    for (const auto& runDirEntry : fs::directory_iterator(runsRoot)) {
        if (!runDirEntry.is_directory()) {
            continue;
        }
        const std::string runId = runDirEntry.path().filename().string();
        for (const auto& fileEntry : fs::recursive_directory_iterator(runDirEntry.path())) {
            if (!fileEntry.is_regular_file()) {
                continue;
            }
            ArtifactEntry row;
            row.runId = runId;
            row.path = fileEntry.path().string();
            row.fileName = fileEntry.path().filename().string();
            row.artifactId = runId + ":" + row.fileName;
            row.kind = kindFromPath(fileEntry.path());
            rows.push_back(row);
        }
    }
    return rows;
}

std::string urlDecode(const std::string& s) {
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

fs::path artifactPathFromId(const fs::path& repoRoot, const std::string& artifactId, std::string& fileNameOut) {
    const auto split = artifactId.find(':');
    if (artifactId.empty() || split == std::string::npos) {
        throw std::runtime_error("invalid artifactId");
    }

    const std::string runId = artifactId.substr(0, split);
    const std::string fileName = artifactId.substr(split + 1);
    if (runId.find("..") != std::string::npos || fileName.find("..") != std::string::npos ||
        runId.find('/') != std::string::npos || runId.find('\\') != std::string::npos ||
        fileName.find('/') != std::string::npos || fileName.find('\\') != std::string::npos) {
        throw std::runtime_error("invalid artifactId path");
    }

    fileNameOut = fileName;
    const fs::path path = repoRoot / "fractal_studio" / "runtime" / "runs" / runId / fileName;
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        throw std::runtime_error("artifact not found");
    }
    return path;
}

} // namespace

std::string artifactsListRoute(const std::filesystem::path& repoRoot, const std::string& query) {
    const std::string kindFilter = urlDecode(getQueryParam(query, "kind"));
    const std::string runIdFilter = urlDecode(getQueryParam(query, "runId"));

    const auto rows = collectArtifacts(repoRoot);

    std::ostringstream ss;
    ss << "{\"items\":[";
    bool first = true;
    for (const auto& row : rows) {
        if (!kindFilter.empty() && row.kind != kindFilter) {
            continue;
        }
        if (!runIdFilter.empty() && row.runId != runIdFilter) {
            continue;
        }
        if (!first) {
            ss << ",";
        }
        first = false;
        ss << "{"
           << "\"artifactId\":\"" << jsonEscape(row.artifactId) << "\","
           << "\"runId\":\"" << jsonEscape(row.runId) << "\","
           << "\"name\":\"" << jsonEscape(row.fileName) << "\","
           << "\"kind\":\"" << jsonEscape(row.kind) << "\","
           << "\"downloadPath\":\"/api/artifacts/download?artifactId=" << jsonEscape(row.artifactId) << "\""
           << "}";
    }
    ss << "]}";
    return ss.str();
}

std::string artifactDownloadBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType, std::string& downloadName) {
    const std::string artifactId = urlDecode(getQueryParam(query, "artifactId"));
    std::string fileName;
    const fs::path path = artifactPathFromId(repoRoot, artifactId, fileName);

    contentType = contentTypeFromPath(path);
    downloadName = fileName;
    return readFileText(path);
}

std::string artifactContentBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType) {
    const std::string artifactId = urlDecode(getQueryParam(query, "artifactId"));
    std::string fileName;
    const fs::path path = artifactPathFromId(repoRoot, artifactId, fileName);

    contentType = contentTypeFromPath(path);
    return readFileText(path);
}

} // namespace fsd
