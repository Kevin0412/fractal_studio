#include "routes.hpp"

#include "adapters.hpp"

#include <cmath>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fsd {

namespace {

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

void validate(const MapRenderParams& p) {
    if (!(p.scale > 0.0) || !std::isfinite(p.scale)) {
        throw std::runtime_error("invalid scale: must be > 0");
    }
    if (!std::isfinite(p.centerRe) || !std::isfinite(p.centerIm)) {
        throw std::runtime_error("invalid center: non-finite values");
    }
    if (p.width < 64 || p.width > 2048) {
        throw std::runtime_error("invalid width: must be 64..2048");
    }
    if (p.height < 64 || p.height > 2048) {
        throw std::runtime_error("invalid height: must be 64..2048");
    }
    if (p.variety < 0 || p.variety > 32) {
        throw std::runtime_error("invalid variety: must be 0..32");
    }
}

} // namespace

std::string mapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    MapRenderParams params{};
    params.centerRe = getNumberField(body, "centerRe").value_or(0.0);
    params.centerIm = getNumberField(body, "centerIm").value_or(0.0);
    params.scale = getNumberField(body, "scale").value_or(4.0);
    params.width = getIntField(body, "width").value_or(1024);
    params.height = getIntField(body, "height").value_or(768);
    params.variety = getIntField(body, "variety").value_or(0);
    params.iterations = getIntField(body, "iterations").value_or(1024);
    params.colorMap = "classic";

    validate(params);

    auto run = runner.createRun("explorer-map");
    runner.setStatus(run.id, "running");

    try {
        const Artifact artifact = runManagedMapRender(repoRoot, run.outputDir, params);
        runner.addArtifact(run.id, artifact);
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    run = runner.getRun(run.id);
    const std::string artifactId = run.id + ":map.png";

    std::ostringstream ss;
    ss << "{"
       << "\"runId\":\"" << run.id << "\","
       << "\"status\":\"" << run.status << "\","
       << "\"artifactId\":\"" << artifactId << "\","
       << "\"imagePath\":\"/api/artifacts/content?artifactId=" << artifactId << "\","
       << "\"effective\":{"
       << "\"centerRe\":" << params.centerRe << ","
       << "\"centerIm\":" << params.centerIm << ","
       << "\"scale\":" << params.scale << ","
       << "\"width\":" << params.width << ","
       << "\"height\":" << params.height << ","
       << "\"variety\":" << params.variety
       << "},"
       << "\"notes\":{\"iterationsApplied\":false,\"colorMapApplied\":false}"
       << "}";
    return ss.str();
}

} // namespace fsd
