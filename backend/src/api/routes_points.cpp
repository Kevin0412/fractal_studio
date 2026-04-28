// routes_points.cpp — native Newton solver endpoint.

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/newton/mandelbrot_sp.hpp"
#include "../compute/special_points.hpp"

#include <complex>
#include <stdexcept>

namespace fsd {

namespace {

Json orbitToJson(const compute::OrbitClassification& o) {
    return Json{
        {"found_repeat", o.found_repeat},
        {"is_center", o.is_center},
        {"is_misiurewicz", o.is_misiurewicz},
        {"preperiod", o.preperiod},
        {"period", o.period},
        {"repeat_error", o.repeat_error},
    };
}

Json variantsToJson(const std::vector<compute::VariantExistence>& variants) {
    Json arr = Json::array();
    for (const auto& v : variants) {
        arr.push_back({
            {"variant_name", v.variant_name},
            {"exists", v.exists},
            {"same_orbit_as_mandelbrot", v.same_orbit_as_mandelbrot},
            {"actual_preperiod", v.actual_preperiod},
            {"actual_period", v.actual_period},
            {"repeat_error", v.repeat_error},
            {"reason", v.reason},
        });
    }
    return arr;
}

Json pointToJson(const compute::SpecialPointResult& p) {
    return Json{
        {"id", p.id},
        {"kind", compute::special_point_kind_name(p.kind)},
        {"preperiod", p.preperiod},
        {"period", p.period},
        {"re", p.re},
        {"im", p.im},
        {"real", p.re},
        {"imag", p.im},
        {"converged", p.converged},
        {"accepted", p.accepted},
        {"visible", p.visible},
        {"residual", p.residual},
        {"newtonIterations", p.newton_iterations},
        {"actual", orbitToJson(p.actual)},
        {"variants", variantsToJson(p.variants)},
        {"reason", p.reason},
    };
}

Json enumResponseToJson(const compute::SpecialPointEnumResponse& r) {
    Json points = Json::array();
    for (const auto& p : r.points) points.push_back(pointToJson(p));
    Json rejected = Json::array();
    for (const auto& p : r.rejected_debug) rejected.push_back(pointToJson(p));
    return Json{
        {"complete", r.complete},
        {"status", r.status},
        {"acceptedCount", r.accepted_count},
        {"expectedCount", r.expected_count},
        {"seedCount", r.seed_count},
        {"newtonSuccessCount", r.newton_success_count},
        {"rejectedCount", r.rejected_count},
        {"points", points},
        {"rejected_debug", rejected},
        {"warning", r.warning},
    };
}

compute::SpecialPointKind parseKind(const Json& j) {
    const std::string raw = j.value("kind", std::string("center"));
    if (raw == "center" || raw == "hyperbolic" || raw == "hyperbolic_center") {
        return compute::SpecialPointKind::HyperbolicCenter;
    }
    if (raw == "misiurewicz") return compute::SpecialPointKind::Misiurewicz;
    throw HttpError(400, Json{{"error", "invalid special point kind"}}.dump());
}

compute::SpecialPointEnumRequest parseEnumRequest(const Json& j) {
    compute::SpecialPointEnumRequest req;
    req.kind = parseKind(j);
    req.period_min = j.value("periodMin", 1);
    req.period_max = j.value("periodMax", req.kind == compute::SpecialPointKind::HyperbolicCenter ? 8 : 4);
    req.preperiod_min = j.value("preperiodMin", 1);
    req.preperiod_max = j.value("preperiodMax", 4);
    req.misiurewicz_period_min = j.value("periodMin", 1);
    req.misiurewicz_period_max = j.value("periodMax", 4);
    req.max_newton_iter = j.value("maxNewtonIter", 60);
    req.max_seed_batches = j.value("maxSeedBatches", 80);
    req.seeds_per_batch = j.value("seedsPerBatch", 2048);
    req.newton_eps = j.value("newtonEps", 1e-13);
    req.classify_eps = j.value("classifyEps", 1e-10);
    req.root_merge_eps = j.value("rootMergeEps", 1e-9);
    req.include_variant_existence = j.value("includeVariantExistence", true);
    req.include_rejected_debug = j.value("includeRejectedDebug", false);
    req.visible_only = j.value("visibleOnly", false);

    if (j.contains("viewport") && j["viewport"].is_object()) {
        const Json& v = j["viewport"];
        req.viewport.enabled = true;
        req.viewport.center_re = v.value("centerRe", -0.75);
        req.viewport.center_im = v.value("centerIm", 0.0);
        req.viewport.scale = v.value("scale", 3.0);
        req.viewport.width = v.value("width", 1200);
        req.viewport.height = v.value("height", 800);
    }
    return req;
}

int totalExpectedOrThrow(const compute::SpecialPointEnumRequest& req) {
    int total = 0;
    if (req.kind == compute::SpecialPointKind::HyperbolicCenter) {
        for (int p = req.period_min; p <= req.period_max; ++p) {
            const int count = compute::expected_center_count(p);
            if (count < 0) throw HttpError(400, Json{{"error", "expected count unavailable"}}.dump());
            total += count;
        }
    } else {
        for (int m = req.preperiod_min; m <= req.preperiod_max; ++m) {
            for (int p = req.misiurewicz_period_min; p <= req.misiurewicz_period_max; ++p) {
                const int count = compute::expected_misiurewicz_count(m, p);
                if (count < 0) {
                    throw HttpError(400, Json{
                        {"error", "expected count unavailable for requested Misiurewicz parameter"},
                        {"suggestion", "reduce preperiod/period range"},
                    }.dump());
                }
                total += count;
            }
        }
    }
    return total;
}

void validateEnumRequest(const compute::SpecialPointEnumRequest& req) {
    if (req.kind == compute::SpecialPointKind::HyperbolicCenter) {
        if (req.period_min < 1 || req.period_max < req.period_min || req.period_max > 10) {
            throw HttpError(400, Json{{"error", "invalid center period range"}, {"limit", "1..10"}}.dump());
        }
    } else {
        if (req.preperiod_min < 1 || req.preperiod_max < req.preperiod_min || req.preperiod_max > 6 ||
            req.misiurewicz_period_min < 1 || req.misiurewicz_period_max < req.misiurewicz_period_min ||
            req.misiurewicz_period_max > 6 ||
            req.preperiod_max + req.misiurewicz_period_max > 10) {
            throw HttpError(400, Json{{"error", "invalid Misiurewicz range"}, {"limit", "preperiod 1..6, period 1..6, preperiod+period <= 10"}}.dump());
        }
    }
    if (req.max_newton_iter < 1 || req.max_newton_iter > 80 ||
        req.max_seed_batches < 1 || req.max_seed_batches > 200 ||
        req.seeds_per_batch < 1 || req.seeds_per_batch > 10000) {
        throw HttpError(400, Json{{"error", "invalid solver limits"}}.dump());
    }
    const int expected = totalExpectedOrThrow(req);
    if (expected > 3000) {
        throw HttpError(400, Json{
            {"error", "parameter range too large"},
            {"expectedCount", expected},
            {"limit", 3000},
            {"suggestion", "reduce period/preperiod range"},
        }.dump());
    }
}

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

std::string specialPointsEnumerateRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    const Json j = parseJsonBody(body);
    compute::SpecialPointEnumRequest req = parseEnumRequest(j);
    validateEnumRequest(req);

    auto run = runner.createRun("special-points-enumerate", body);
    runner.setStatus(run.id, "running");
    runner.setCancelable(run.id, true);

    auto setProgress = [&](const std::string& stage, int accepted, int expected, int seeds, int batch) {
        Json progress = {
            {"taskType", "special_points"},
            {"stage", stage},
            {"acceptedCount", accepted},
            {"expectedCount", expected},
            {"seedCount", seeds},
            {"batchIndex", batch},
            {"elapsedMs", runner.runElapsedMs(run.id)},
            {"cancelable", true},
        };
        runner.setProgress(run.id, progress.dump());
    };

    setProgress("enumerating", 0, totalExpectedOrThrow(req), 0, 0);
    try {
        compute::SpecialPointEnumResponse resp = compute::enumerate_special_points(
            req,
            [&](int taskIndex, int taskCount, int accepted, int expected, int seeds, int batch) {
                Json progress = {
                    {"taskType", "special_points"},
                    {"stage", "enumerating"},
                    {"taskIndex", taskIndex},
                    {"taskCount", taskCount},
                    {"acceptedCount", accepted},
                    {"expectedCount", expected},
                    {"seedCount", seeds},
                    {"batchIndex", batch},
                    {"elapsedMs", runner.runElapsedMs(run.id)},
                    {"cancelable", true},
                };
                runner.setProgress(run.id, progress.dump());
                return !runner.isCancelRequested(run.id);
            });

        if (runner.isCancelRequested(run.id) || resp.status == "cancelled") {
            runner.setStatus(run.id, "cancelled");
            return Json{{"runId", run.id}, {"status", "cancelled"}}.dump();
        }

        Json responseJson = enumResponseToJson(resp);
        responseJson["runId"] = run.id;

        Json artifact = {
            {"version", 1},
            {"timestamp", nowIso8601()},
            {"request", j},
            {"response", responseJson},
            {"points", responseJson["points"]},
        };
        if (req.include_rejected_debug) artifact["rejected_debug"] = responseJson["rejected_debug"];
        const std::filesystem::path outPath = std::filesystem::path(run.outputDir) / "special_points.json";
        atomicWriteText(outPath, artifact.dump(2));
        runner.addArtifact(run.id, Artifact{"special-points", outPath.string(), "report"});

        setProgress(resp.status, resp.accepted_count, resp.expected_count, resp.seed_count, req.max_seed_batches);
        runner.setStatus(run.id, "completed");
        const std::string artId = run.id + ":special_points.json";
        responseJson["reportArtifactId"] = artId;
        responseJson["reportDownloadUrl"] = "/api/artifacts/download?artifactId=" + artId;
        return responseJson.dump();
    } catch (const std::exception& e) {
        if (runner.isCancelRequested(run.id)) {
            runner.setStatus(run.id, "cancelled");
        } else {
            runner.setStatus(run.id, "failed");
            setProgress("failed", 0, totalExpectedOrThrow(req), 0, 0);
        }
        throw;
    }
}

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
