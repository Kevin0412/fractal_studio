#include "routes.hpp"
#include "adapters.hpp"
#include "report.hpp"
#include "system_checks.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fsd {

std::string systemCheckRoute() {
    const auto openmp = checkOpenMP();
    const auto cuda = checkCudaRuntime();
    std::ostringstream ss;
    ss << "{\"openmp\":" << (openmp ? "true" : "false") << ",\"cuda\":" << (cuda ? "true" : "false") << "}";
    return ss.str();
}

std::string moduleRoute(const std::string& moduleName, JobRunner& runner, const std::filesystem::path& repoRoot) {
    auto run = runner.createRun(moduleName);
    runner.setStatus(run.id, "running");

    auto addConclusionArtifact = [&](const RunRecord& finalRun) {
        const std::filesystem::path conclusionPath = std::filesystem::path(finalRun.outputDir) / "conclusions.json";
        std::ofstream out(conclusionPath);
        out << buildConclusions(finalRun) << "\n";
        runner.addArtifact(finalRun.id, Artifact{"conclusions", conclusionPath.string(), "report"});
    };

    try {
        if (moduleName == "atlas") {
            auto artifact = runLegacyVariantAtlas(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "special-points") {
            auto artifact = runLegacySpecialPoints(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "transition-conversion") {
            auto artifact = runLegacyTransition(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "hidden-structure") {
            auto artifact = runLegacyHiddenStructure(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "hidden-structure-family") {
            auto artifact = runLegacyHiddenStructureFamily(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "stl-export") {
            auto artifact = runLegacyStlExport(repoRoot, run.outputDir);
            runner.addArtifact(run.id, artifact);
        } else if (moduleName == "conclusions") {
            addConclusionArtifact(runner.getRun(run.id));
        } else if (moduleName == "run-all") {
            const std::vector<std::string> modules = {
                "atlas",
                "special-points",
                "transition-conversion",
                "hidden-structure",
                "hidden-structure-family",
                "stl-export"
            };

            for (const auto& m : modules) {
                if (m == "atlas") {
                    runner.addArtifact(run.id, runLegacyVariantAtlas(repoRoot, run.outputDir));
                } else if (m == "special-points") {
                    runner.addArtifact(run.id, runLegacySpecialPoints(repoRoot, run.outputDir));
                } else if (m == "transition-conversion") {
                    runner.addArtifact(run.id, runLegacyTransition(repoRoot, run.outputDir));
                } else if (m == "hidden-structure") {
                    runner.addArtifact(run.id, runLegacyHiddenStructure(repoRoot, run.outputDir));
                } else if (m == "hidden-structure-family") {
                    runner.addArtifact(run.id, runLegacyHiddenStructureFamily(repoRoot, run.outputDir));
                } else if (m == "stl-export") {
                    runner.addArtifact(run.id, runLegacyStlExport(repoRoot, run.outputDir));
                }
            }
            addConclusionArtifact(runner.getRun(run.id));
        } else {
            throw std::runtime_error("unknown module: " + moduleName);
        }
        runner.setStatus(run.id, "completed");
    } catch (const std::exception& ex) {
        runner.setStatus(run.id, "failed");
        std::ostringstream err;
        err << "{\"runId\":\"" << run.id << "\",\"status\":\"failed\",\"error\":\"" << ex.what() << "\"}";
        return err.str();
    }

    run = runner.getRun(run.id);
    std::ostringstream ss;
    ss << "{\"runId\":\"" << run.id << "\",\"status\":\"" << run.status << "\",\"artifactCount\":" << run.artifacts.size() << "}";
    return ss.str();
}

} // namespace fsd
