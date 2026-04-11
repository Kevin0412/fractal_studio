// routes_map.cpp
//
// Native map renderer endpoint. All 10 variants × all metrics × optional
// Mandelbrot↔Burning-Ship transition. Dispatches to compute/map_kernel.cpp or
// compute/transition_kernel.cpp and writes a PNG via OpenCV.

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/map_kernel.hpp"
#include "../compute/tile_scheduler.hpp"
#include "../compute/transition_kernel.hpp"
#include "../compute/image_io.hpp"
#include "../compute/variants.hpp"
#include "../compute/escape_time.hpp"
#include "../compute/colormap.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace fsd {

namespace {

compute::Variant parseVariant(const std::string& s) {
    compute::Variant v;
    if (compute::variant_from_name(s.c_str(), v)) return v;
    // Backwards compat with old "variety" integer.
    try {
        int i = std::stoi(s);
        if (i >= 0 && i <= 9) return static_cast<compute::Variant>(i);
    } catch (...) {}
    return compute::Variant::Mandelbrot;
}

compute::Metric parseMetric(const std::string& s) {
    compute::Metric m;
    if (compute::metric_from_name(s.c_str(), m)) return m;
    return compute::Metric::Escape;
}

compute::Colormap parseColormap(const std::string& s) {
    compute::Colormap c;
    if (compute::colormap_from_name(s.c_str(), c)) return c;
    return compute::Colormap::ClassicCos;
}

} // namespace

std::string mapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    (void)repoRoot;
    const Json j = parseJsonBody(body);

    const double cRe     = j.value("centerRe", -0.75);
    const double cIm     = j.value("centerIm",  0.0);
    const double scale   = j.value("scale",     3.0);
    const int width      = j.value("width",     1024);
    const int height     = j.value("height",     768);
    const int iters      = j.value("iterations", 1024);
    const double bailout = j.value("bailout",   2.0);

    const std::string variantStr  = j.value("variant",  std::string("mandelbrot"));
    const std::string metricStr   = j.value("metric",   std::string("escape"));
    const std::string colormapStr = j.value("colorMap", std::string("classic_cos"));
    const bool julia              = j.value("julia",    false);
    const double juliaRe          = j.value("juliaRe",  0.0);
    const double juliaIm          = j.value("juliaIm",  0.0);

    const bool hasTheta = j.contains("transitionTheta") && !j["transitionTheta"].is_null();
    const double theta  = hasTheta ? j.value("transitionTheta", 0.0) : 0.0;

    const std::string scalarType = j.value("scalarType", std::string("auto"));
    const std::string engine     = j.value("engine",     std::string("openmp"));
    const bool smooth            = j.value("smooth",     false);

    // Basic validation.
    if (!(scale > 0.0) || !std::isfinite(scale))            throw std::runtime_error("invalid scale");
    if (width < 64 || width  > 4096)                        throw std::runtime_error("invalid width");
    if (height < 64 || height > 4096)                       throw std::runtime_error("invalid height");
    if (iters < 1 || iters > 1000000)                       throw std::runtime_error("invalid iterations");
    if (!std::isfinite(cRe) || !std::isfinite(cIm))         throw std::runtime_error("invalid center");
    if (hasTheta && (theta < 0.0 || theta > 3.15))          throw std::runtime_error("invalid transitionTheta (0..π)");

    auto run = runner.createRun("map", body);
    runner.setStatus(run.id, "running");

    std::string artifactName;
    std::string scalarUsed = "fp64";
    std::string engineUsed = "openmp";
    double elapsed = 0.0;

    try {
        cv::Mat out;
        if (hasTheta) {
            compute::TransitionParams p;
            p.center_re = cRe;
            p.center_im = cIm;
            p.scale     = scale;
            p.width     = width;
            p.height    = height;
            p.iterations = iters;
            p.bailout    = bailout;
            p.theta      = theta;
            p.metric       = parseMetric(metricStr);
            p.colormap     = parseColormap(colormapStr);
            p.smooth       = smooth;
            p.pairwise_cap = j.value("pairwiseCap", 64);
            auto stats = compute::render_transition(p, out);
            elapsed = stats.elapsed_ms;
            artifactName = "transition.png";
        } else {
            compute::MapParams p;
            p.center_re  = cRe;
            p.center_im  = cIm;
            p.scale      = scale;
            p.width      = width;
            p.height     = height;
            p.iterations = iters;
            p.bailout    = bailout;
            p.variant    = parseVariant(variantStr);
            p.metric     = parseMetric(metricStr);
            p.colormap   = parseColormap(colormapStr);
            p.julia      = julia;
            p.julia_re   = juliaRe;
            p.julia_im   = juliaIm;
            p.scalar_type = scalarType;
            p.engine      = engine;
            p.smooth      = smooth;

            if (engine == "hybrid") {
                auto stats = compute::render_map_hybrid(p, out);
                elapsed = stats.total_ms;
                scalarUsed = stats.scalar_used;
                engineUsed = stats.engine_used;
            } else {
                auto stats = compute::render_map(p, out);
                elapsed = stats.elapsed_ms;
                scalarUsed = stats.scalar_used;
                engineUsed = stats.engine_used;
            }
            artifactName = "map.png";
        }

        const std::filesystem::path imagePath =
            std::filesystem::path(run.outputDir) / artifactName;
        compute::write_png(imagePath.string(), out);
        runner.addArtifact(run.id, Artifact{"map", imagePath.string(), "image"});
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    const std::string artifactId = run.id + ":" + artifactName;
    Json resp = {
        {"runId",    run.id},
        {"status",   "completed"},
        {"artifactId", artifactId},
        {"imagePath", "/api/artifacts/content?artifactId=" + artifactId},
        {"generatedMs", elapsed},
        {"width",  width},
        {"height", height},
        {"scalarUsed", scalarUsed},
        {"engineUsed", engineUsed},
        {"effective", {
            {"centerRe",  cRe},
            {"centerIm",  cIm},
            {"scale",     scale},
            {"iterations", iters},
            {"variant",   variantStr},
            {"metric",    metricStr},
            {"colorMap",  colormapStr},
            {"julia",     julia},
            {"juliaRe",   juliaRe},
            {"juliaIm",   juliaIm},
            {"transitionTheta", hasTheta ? theta : 0.0},
            {"transitionActive", hasTheta},
        }},
    };
    return resp.dump();
}

} // namespace fsd
