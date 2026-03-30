#include "hardware_probe.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace fsd {

namespace {

std::string trim(const std::string& input) {
    size_t start = 0;
    while (start < input.size() && (input[start] == ' ' || input[start] == '\t' || input[start] == '\n' || input[start] == '\r')) {
        ++start;
    }

    size_t end = input.size();
    while (end > start && (input[end - 1] == ' ' || input[end - 1] == '\t' || input[end - 1] == '\n' || input[end - 1] == '\r')) {
        --end;
    }

    return input.substr(start, end - start);
}

std::string readCommandOutput(const std::string& command) {
    std::array<char, 256> buffer{};
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return "";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    pclose(pipe);
    return trim(output);
}

std::string readCpuModel() {
    std::ifstream cpuInfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuInfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            const size_t pos = line.find(':');
            if (pos != std::string::npos) {
                return trim(line.substr(pos + 1));
            }
        }
    }
    return "unknown";
}

int readPhysicalCoreCount() {
    const std::string output = readCommandOutput("bash -lc \"lscpu | awk -F: '/Core\\(s\\) per socket/ {gsub(/^[ \\t]+/,\\\"\\\",\\$2); print \\$2}'\"");
    const std::string sockets = readCommandOutput("bash -lc \"lscpu | awk -F: '/Socket\\(s\\)/ {gsub(/^[ \\t]+/,\\\"\\\",\\$2); print \\$2}'\"");

    if (output.empty() || sockets.empty()) {
        return static_cast<int>(std::thread::hardware_concurrency());
    }

    return std::stoi(output) * std::stoi(sockets);
}

std::pair<long long, long long> readMemoryKiB() {
    std::ifstream memInfo("/proc/meminfo");
    std::string line;
    long long totalKiB = 0;
    long long availableKiB = 0;

    while (std::getline(memInfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            std::istringstream in(line.substr(9));
            in >> totalKiB;
        }
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream in(line.substr(13));
            in >> availableKiB;
        }
    }

    return {totalKiB, availableKiB};
}

std::pair<std::string, std::string> readGpuInfo() {
    const std::string name = readCommandOutput("bash -lc \"nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1\"");
    const std::string memory = readCommandOutput("bash -lc \"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1\"");

    if (name.empty()) {
        return {"unavailable", "unavailable"};
    }

    if (memory.empty()) {
        return {name, "unavailable"};
    }

    return {name, memory + " MiB"};
}

} // namespace

std::string systemHardwareRoute() {
    const std::string cpuModel = readCpuModel();
    const int logicalCores = static_cast<int>(std::thread::hardware_concurrency());
    const int physicalCores = readPhysicalCoreCount();
    const auto memory = readMemoryKiB();
    const auto gpu = readGpuInfo();

    std::ostringstream out;
    out << "{"
        << "\"cpuModel\":\"" << cpuModel << "\","
        << "\"cpuLogicalCores\":" << logicalCores << ","
        << "\"cpuPhysicalCores\":" << physicalCores << ","
        << "\"memoryTotalMiB\":" << (memory.first / 1024) << ","
        << "\"memoryAvailableMiB\":" << (memory.second / 1024) << ","
        << "\"gpuModel\":\"" << gpu.first << "\","
        << "\"gpuMemory\":\"" << gpu.second << "\""
        << "}";
    return out.str();
}

} // namespace fsd
