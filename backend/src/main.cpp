#include "http_server.hpp"
#include "job_runner.hpp"
#include "path_guard.hpp"
#include "routes.hpp"
#include "db.hpp"

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

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

    // Walk up from cwd until we find the repo root (marked by C_mandelbrot/).
    fs::path probe = fs::current_path();
    while (!probe.empty() && !fs::exists(probe / "C_mandelbrot")) {
        const auto parent = probe.parent_path();
        if (parent == probe) break;
        probe = parent;
    }
    const fs::path repoRoot = probe;

    const fs::path runtimeRoot = repoRoot / "fractal_studio" / "runtime";
    const fs::path dbDir = runtimeRoot / "db";
    fs::create_directories(dbDir);
    fsd::Db db(dbDir / "fractal_studio.sqlite3");
    db.ensureSchema();

    fsd::PathGuard guard(repoRoot);
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
