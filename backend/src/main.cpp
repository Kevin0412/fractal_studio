#include "http_server.hpp"
#include "job_runner.hpp"
#include "path_guard.hpp"
#include "routes.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    fs::path probe = fs::current_path();
    while (!probe.empty() && !fs::exists(probe / "C_mandelbrot")) {
        const auto parent = probe.parent_path();
        if (parent == probe) {
            break;
        }
        probe = parent;
    }
    const fs::path repoRoot = probe;

    fsd::PathGuard guard(repoRoot);
    fsd::JobRunner runner(repoRoot / "fractal_studio" / "runtime");

    const auto health = fsd::systemCheckRoute();
    std::cout << "fractal_studio backend skeleton ready" << std::endl;
    std::cout << "health: " << health << std::endl;

    const std::string mode = (argc > 1) ? argv[1] : "atlas";
    if (mode == "serve") {
        int port = 8080;
        if (argc > 2) {
            port = std::stoi(argv[2]);
        }
        fsd::HttpServer server(port, runner, repoRoot);
        server.serveForever();
        return 0;
    }

    const auto run = fsd::moduleRoute(mode, runner, repoRoot);
    std::cout << "run: " << run << std::endl;

    const auto legacyPath = repoRoot / "cfiles" / "should_not_write.txt";
    if (!guard.isWriteAllowed(legacyPath)) {
        std::cout << guard.denyReason(legacyPath) << std::endl;
    }

    return 0;
}
