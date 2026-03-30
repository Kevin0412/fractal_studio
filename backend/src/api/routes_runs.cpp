#include "routes.hpp"

#include <sstream>

namespace fsd {

std::string createRunRoute(JobRunner& runner, const std::string& module) {
    const auto run = runner.createRun(module);
    std::ostringstream ss;
    ss << "{\"runId\":\"" << run.id << "\",\"status\":\"" << run.status << "\",\"outputDir\":\"" << run.outputDir << "\"}";
    return ss.str();
}

std::string getRunRoute(JobRunner& runner, const std::string& runId) {
    const auto run = runner.getRun(runId);
    std::ostringstream ss;
    ss << "{\"runId\":\"" << run.id << "\",\"status\":\"" << run.status << "\",\"artifactCount\":" << run.artifacts.size() << "}";
    return ss.str();
}

} // namespace fsd
