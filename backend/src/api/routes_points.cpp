// routes_points.cpp — native Newton solver endpoint.

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/newton/mandelbrot_sp.hpp"

#include <complex>
#include <stdexcept>

namespace fsd {

namespace {

void persistPoints(
    const std::filesystem::path& repoRoot,
    const std::vector<compute::newton_sp::SolvedPoint>& pts,
    const std::string& sourceMode,
    const std::string& pointType,
    Json& out
) {
    Db db = openDb(repoRoot);
    const std::string createdAt = nowIso8601();
    out = Json::array();
    for (const auto& pt : pts) {
        SpecialPointRecord rec;
        rec.id = makeId();
        rec.family = pt.variant;
        rec.pointType = pointType;
        rec.k = pt.k;
        rec.p = pt.p;
        rec.re = pt.c.real();
        rec.im = pt.c.imag();
        rec.sourceMode = sourceMode;
        rec.createdAt = createdAt;
        db.insertSpecialPoint(rec);
        out.push_back({
            {"id", rec.id},
            {"real", rec.re},
            {"imag", rec.im},
            {"k", rec.k},
            {"p", rec.p},
            {"family", rec.family},
        });
    }
}

} // namespace

std::string specialPointsAutoRoute(const std::filesystem::path& repoRoot, const std::string& body) {
    const Json j = parseJsonBody(body);
    const int k = j.value("k", 1);
    const int p = j.value("p", 1);
    const std::string pointType = j.value("pointType", std::string(k == 0 ? "hyperbolic" : "misiurewicz"));

    if (pointType == "misiurewicz" && k <= 0) throw std::runtime_error("misiurewicz requires k > 0");
    if (pointType == "hyperbolic"  && k != 0) throw std::runtime_error("hyperbolic requires k = 0");
    if (p < 1)                                 throw std::runtime_error("period p must be >= 1");

    const auto pts = compute::newton_sp::auto_solve(k, p);

    Json items;
    persistPoints(repoRoot, pts, "auto", pointType, items);

    Json resp = {
        {"mode", "auto"},
        {"k", k},
        {"p", p},
        {"count", pts.size()},
        {"points", items},
    };
    return resp.dump();
}

std::string specialPointsSeedRoute(const std::filesystem::path& repoRoot, const std::string& body) {
    const Json j = parseJsonBody(body);
    const int k = j.value("k", 1);
    const int p = j.value("p", 1);
    const double sr = j.value("re", 0.0);
    const double si = j.value("im", 0.0);

    if (p < 1) throw std::runtime_error("period p must be >= 1");

    const auto pts = compute::newton_sp::seed_solve(k, p, std::complex<double>(sr, si));

    Json items;
    persistPoints(repoRoot, pts, "seed", "newton", items);

    Json resp = {
        {"mode", "seed"},
        {"k", k},
        {"p", p},
        {"converged", !pts.empty()},
        {"points", items},
    };
    return resp.dump();
}

std::string specialPointsListRoute(const std::filesystem::path& repoRoot, const std::string& query) {
    const std::string family = urlDecode(getQueryParam(query, "family"));
    int k = -1, p = -1;
    const std::string kRaw = getQueryParam(query, "k");
    const std::string pRaw = getQueryParam(query, "p");
    if (!kRaw.empty()) k = std::stoi(kRaw);
    if (!pRaw.empty()) p = std::stoi(pRaw);

    Db db = openDb(repoRoot);
    const auto rows = db.listSpecialPoints(family, k, p);

    Json items = Json::array();
    for (const auto& r : rows) {
        items.push_back({
            {"id",         r.id},
            {"family",     r.family},
            {"pointType",  r.pointType},
            {"k",          r.k},
            {"p",          r.p},
            {"real",       r.re},
            {"imag",       r.im},
            {"sourceMode", r.sourceMode},
            {"createdAt",  r.createdAt},
        });
    }
    Json resp = {{"items", items}};
    return resp.dump();
}

} // namespace fsd
