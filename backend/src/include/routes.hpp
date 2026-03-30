#pragma once

#include "job_runner.hpp"

#include <filesystem>
#include <string>

namespace fsd {

std::string systemCheckRoute();
std::string createRunRoute(JobRunner& runner, const std::string& module);
std::string getRunRoute(JobRunner& runner, const std::string& runId);
std::string moduleRoute(const std::string& moduleName, JobRunner& runner, const std::filesystem::path& repoRoot);

std::string specialPointsAutoRoute(const std::filesystem::path& repoRoot, const std::string& body);
std::string specialPointsSeedRoute(const std::filesystem::path& repoRoot, const std::string& body);
std::string specialPointsListRoute(const std::filesystem::path& repoRoot, const std::string& query);

std::string artifactsListRoute(const std::filesystem::path& repoRoot, const std::string& query);
std::string artifactDownloadBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType, std::string& downloadName);
std::string artifactContentBody(const std::filesystem::path& repoRoot, const std::string& query, std::string& contentType);

std::string mapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body);

} // namespace fsd
