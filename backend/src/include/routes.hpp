#pragma once

#include "job_runner.hpp"

#include <filesystem>
#include <string>

namespace fsd {

// System / hardware
std::string systemCheckRoute();
std::string systemHardwareRoute();

// Map (native): escape/metric/transition all dispatch through here.
// Body is JSON; response is JSON describing the generated artifact.
std::string mapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);

// ln-map renderer (Phase 1 ships this; Phase 2 adds the video exporter that
// consumes its output).
std::string lnMapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);

// Video export (ln-map → mp4). Real implementation in Phase 2.
std::string zoomVideoRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);

// HS heightfield mesh + transition 3D mesh.
std::string hsMeshRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);
std::string transitionMeshRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);
// Voxel grid (Minecraft-style) for the transition renderer.
std::string transitionVoxelsRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);

// Special points (native Newton solver).
std::string specialPointsAutoRoute(const std::filesystem::path& repoRoot, const std::string& body);
std::string specialPointsSeedRoute(const std::filesystem::path& repoRoot, const std::string& body);
std::string specialPointsListRoute(const std::filesystem::path& repoRoot, const std::string& query);

// Benchmark
std::string benchmarkRoute(const std::string& body);

// Runs
std::string runsListRoute(const std::filesystem::path& repoRoot, const std::string& query);

// Artifacts (filesystem scan of runs dir; artifactId = "runId:fileName")
std::string artifactsListRoute(const std::filesystem::path& repoRoot, const std::string& query);
std::string artifactDownloadBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType, std::string& downloadName);
std::string artifactContentBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType);

} // namespace fsd
