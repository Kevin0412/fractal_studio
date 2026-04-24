#include "http_server.hpp"
#include "job_runner.hpp"
#include "routes.hpp"
#include "db.hpp"

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

std::filesystem::path find_studio_root(std::filesystem::path start) {
    namespace fs = std::filesystem;
    start = fs::weakly_canonical(std::move(start));
    for (fs::path probe = start; !probe.empty(); probe = probe.parent_path()) {
        if (fs::exists(probe / "backend" / "CMakeLists.txt") &&
            fs::exists(probe / "frontend" / "package.json")) {
            return probe;
        }
        if (fs::exists(probe / "fractal_studio" / "backend" / "CMakeLists.txt") &&
            fs::exists(probe / "fractal_studio" / "frontend" / "package.json")) {
            return probe / "fractal_studio";
        }
        const auto parent = probe.parent_path();
        if (parent == probe) break;
    }
    throw std::runtime_error("could not locate fractal_studio root");
}

} // namespace

static void crash_handler(int sig) {
    // Write backtrace to stderr (goes to the log file when redirected).
    // async-signal-safe: uses write(), not printf/cout.
    void* buf[64];
    const int n = ::backtrace(buf, 64);
    const char* msg = "=== CRASH HANDLER ===\n";
    ::write(STDERR_FILENO, msg, __builtin_strlen(msg));
    ::backtrace_symbols_fd(buf, n, STDERR_FILENO);
    // Re-raise with default handler so the core dump is still produced.
    std::signal(sig, SIG_DFL);
    ::raise(sig);
}

int main(int argc, char* argv[]) {
    // Catch segfaults and print a backtrace before dying.
    std::signal(SIGSEGV, crash_handler);
    std::signal(SIGABRT, crash_handler);
    std::signal(SIGBUS,  crash_handler);

    // Suppress SIGPIPE so that writing to a closed client socket returns -1
    // instead of crashing the process. Browsers close connections aggressively
    // (page refresh, navigation, prefetch cancellation) and ::send() would
    // otherwise deliver SIGPIPE the moment it tries to respond.
    std::signal(SIGPIPE, SIG_IGN);

    namespace fs = std::filesystem;

    const fs::path studioRoot = find_studio_root(fs::current_path());
    const fs::path repoRoot = studioRoot.parent_path();

    const fs::path runtimeRoot = studioRoot / "runtime";
    const fs::path dbDir = runtimeRoot / "db";
    fs::create_directories(dbDir);
    fsd::Db db(dbDir / "fractal_studio.sqlite3");
    db.ensureSchema();

    fsd::JobRunner runner(runtimeRoot, &db);

    std::cout << "fractal_studio backend ready (native compute)" << std::endl;
    std::cout << "health: " << fsd::systemCheckRoute() << std::endl;

    int port = 18080;  // frontend api.ts default
    if (argc > 1) {
        // First positional arg can be either "serve" (legacy) or a port number.
        const std::string arg1 = argv[1];
        if (arg1 == "serve") {
            if (argc > 2) port = std::stoi(argv[2]);
        } else {
            try { port = std::stoi(arg1); } catch (...) {}
        }
    }

    fsd::HttpServer server(port, runner, repoRoot);
    server.serveForever();
    return 0;
}
