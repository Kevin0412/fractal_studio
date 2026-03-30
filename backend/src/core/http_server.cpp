#include "http_server.hpp"

#include "hardware_probe.hpp"
#include "routes.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace fsd {

HttpServer::HttpServer(int port, JobRunner& runner, std::filesystem::path repoRoot)
    : port_(port), runner_(runner), repoRoot_(std::move(repoRoot)) {}

std::string HttpServer::makeHttpResponse(int status, const std::string& body, const std::string& contentType, const std::string& extraHeaders) {
    const char* statusText = "OK";
    if (status == 404) statusText = "Not Found";
    if (status == 405) statusText = "Method Not Allowed";
    if (status == 500) statusText = "Internal Server Error";

    std::ostringstream ss;
    ss << "HTTP/1.1 " << status << " " << statusText << "\r\n"
       << "Content-Type: " << contentType << "\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
       << "Access-Control-Allow-Headers: Content-Type\r\n"
       << extraHeaders
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    return ss.str();
}

std::string HttpServer::handleRequest(const std::string& request) const {
    try {
    auto splitPathAndQuery = [](const std::string& rawPath) {
        const std::size_t q = rawPath.find('?');
        if (q == std::string::npos) {
            return std::make_tuple(rawPath, std::string());
        }
        return std::make_tuple(rawPath.substr(0, q), rawPath.substr(q + 1));
    };

    const std::size_t headerEnd = request.find("\r\n\r\n");
    const std::string head = (headerEnd == std::string::npos) ? request : request.substr(0, headerEnd);
    const std::string body = (headerEnd == std::string::npos) ? std::string() : request.substr(headerEnd + 4);

    std::istringstream in(head);
    std::string method;
    std::string rawPath;
    std::string version;
    in >> method >> rawPath >> version;

    const auto [path, query] = splitPathAndQuery(rawPath);

    if (method == "OPTIONS") {
        return makeHttpResponse(200, "{}");
    }

    if (method == "GET" && path == "/api/system/check") {
        return makeHttpResponse(200, systemCheckRoute());
    }

    if (method == "GET" && path == "/api/system/hardware") {
        return makeHttpResponse(200, systemHardwareRoute());
    }

    if (method == "POST" && path.rfind("/api/modules/", 0) == 0) {
        const std::string module = path.substr(std::strlen("/api/modules/"));
        return makeHttpResponse(200, moduleRoute(module, runner_, repoRoot_));
    }

    if (method == "POST" && path == "/api/special-points/auto") {
        return makeHttpResponse(200, specialPointsAutoRoute(repoRoot_, body));
    }

    if (method == "POST" && path == "/api/map/render") {
        return makeHttpResponse(200, mapRenderRoute(repoRoot_, runner_, body));
    }

    if (method == "POST" && path == "/api/special-points/seed") {
        return makeHttpResponse(200, specialPointsSeedRoute(repoRoot_, body));
    }

    if (method == "GET" && path == "/api/special-points") {
        return makeHttpResponse(200, specialPointsListRoute(repoRoot_, query));
    }

    if (method == "GET" && path == "/api/artifacts") {
        return makeHttpResponse(200, artifactsListRoute(repoRoot_, query));
    }

    if (method == "GET" && path == "/api/artifacts/download") {
        try {
            std::string contentType;
            std::string downloadName;
            const std::string bodyText = artifactDownloadBody(repoRoot_, query, contentType, downloadName);
            const std::string headers = "Content-Disposition: attachment; filename=\"" + downloadName + "\"\r\n";
            return makeHttpResponse(200, bodyText, contentType, headers);
        } catch (const std::exception& ex) {
            return makeHttpResponse(404, std::string("{\"error\":\"") + ex.what() + "\"}");
        }
    }

    if (method == "GET" && path == "/api/artifacts/content") {
        try {
            std::string contentType;
            const std::string bodyText = artifactContentBody(repoRoot_, query, contentType);
            return makeHttpResponse(200, bodyText, contentType);
        } catch (const std::exception& ex) {
            return makeHttpResponse(404, std::string("{\"error\":\"") + ex.what() + "\"}");
        }
    }

    if (method != "GET" && method != "POST") {
        return makeHttpResponse(405, "{\"error\":\"method not allowed\"}");
    }

    return makeHttpResponse(404, "{\"error\":\"not found\"}");
    } catch (const std::exception& ex) {
        return makeHttpResponse(500, std::string("{\"error\":\"") + ex.what() + "\"}");
    }
}

void HttpServer::serveForever() {
    const int serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) {
        throw std::runtime_error("failed to create server socket");
    }

    int opt = 1;
    ::setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(serverFd);
        throw std::runtime_error("failed to bind server socket");
    }

    if (::listen(serverFd, 16) < 0) {
        ::close(serverFd);
        throw std::runtime_error("failed to listen");
    }

    std::cout << "HTTP server listening on 0.0.0.0:" << port_ << std::endl;

    while (true) {
        const int clientFd = ::accept(serverFd, nullptr, nullptr);
        if (clientFd < 0) {
            continue;
        }

        char buffer[16384];
        const ssize_t n = ::recv(clientFd, buffer, sizeof(buffer) - 1, 0);
        if (n > 0) {
            buffer[n] = '\0';
            const std::string req(buffer);
            const std::string resp = handleRequest(req);
            ::send(clientFd, resp.c_str(), resp.size(), 0);
        }
        ::close(clientFd);
    }
}

} // namespace fsd
