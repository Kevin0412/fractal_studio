// routes_mesh.cpp
//
// Real implementations of:
//   POST /api/hs/mesh         — heightfield mesh for any HS metric × variant
//   POST /api/transition/mesh — marching-cubes mesh of the 3D transition set

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/hs/heightfield_mesh.hpp"
#include "../compute/transition_volume.hpp"
#include "../compute/marching_cubes.hpp"
#include "../compute/mesh_io.hpp"
#include "../compute/variants.hpp"
#include "../compute/escape_time.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace fsd {

namespace {

compute::Variant parseVariant(const std::string& s) {
    compute::Variant v;
    if (compute::variant_from_name(s.c_str(), v)) return v;
    return compute::Variant::Mandelbrot;
}

compute::Metric parseMetric(const std::string& s) {
    compute::Metric m;
    if (compute::metric_from_name(s.c_str(), m)) return m;
    return compute::Metric::MinAbs;
}

} // namespace

std::string hsMeshRoute(const std::filesystem::path&, JobRunner& runner, const std::string& body) {
    const Json j = parseJsonBody(body);

    compute::hs::HsMeshParams p;
    p.center_re = j.value("centerRe",  -0.75);
    p.center_im = j.value("centerIm",    0.0);
    p.scale     = j.value("scale",       3.0);
    p.resolution = j.value("resolution", 192);
    p.iterations = j.value("iterations", 512);
    p.bailout    = j.value("bailout",    2.0);
    p.heightScale = j.value("heightScale", 0.6);
    p.heightClamp = j.value("heightClamp", 2.0);
    p.variant = parseVariant(j.value("variant", std::string("mandelbrot")));
    p.metric  = parseMetric (j.value("metric",  std::string("min_abs")));

    if (p.resolution < 8 || p.resolution > 4096) throw std::runtime_error("invalid resolution");
    if (p.iterations < 1 || p.iterations > 1000000) throw std::runtime_error("invalid iterations");

    auto run = runner.createRun("hs-mesh", body);
    runner.setStatus(run.id, "running");

    double elapsed = 0.0;
    size_t vc = 0, tc = 0;

    try {
        const auto t0 = std::chrono::steady_clock::now();
        compute::Mesh mesh = compute::hs::buildHsMesh(p);
        const auto t1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        vc = mesh.vertices.size();
        tc = mesh.triangleCount();

        const std::filesystem::path glbPath =
            std::filesystem::path(run.outputDir) / "hs_mesh.glb";
        const std::filesystem::path stlPath =
            std::filesystem::path(run.outputDir) / "hs_mesh.stl";
        compute::writeGlb(glbPath.string(), mesh);
        compute::writeStlBinary(stlPath.string(), mesh);

        runner.addArtifact(run.id, Artifact{"hs-mesh", glbPath.string(), "mesh"});
        runner.addArtifact(run.id, Artifact{"hs-mesh", stlPath.string(), "stl"});
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    const std::string glbId = run.id + ":hs_mesh.glb";
    const std::string stlId = run.id + ":hs_mesh.stl";
    Json resp = {
        {"runId",      run.id},
        {"status",     "completed"},
        {"glbArtifactId", glbId},
        {"stlArtifactId", stlId},
        {"glbUrl",     "/api/artifacts/content?artifactId=" + glbId},
        {"stlUrl",     "/api/artifacts/download?artifactId=" + stlId},
        {"vertexCount", vc},
        {"triangleCount", tc},
        {"generatedMs", elapsed},
    };
    return resp.dump();
}

// Transition 3D volume mesh (Mandelbrot ↔ Burning Ship bridge as a 3D object).
std::string transitionMeshRoute(const std::filesystem::path&, JobRunner& runner, const std::string& body) {
    const Json j = parseJsonBody(body);

    compute::TransitionVolumeParams p;
    p.centerX   = j.value("centerX",   0.0);
    p.centerY   = j.value("centerY",   0.0);
    p.centerZ   = j.value("centerZ",   0.0);
    p.extent    = j.value("extent",    2.0);
    p.resolution = j.value("resolution", 96);
    p.iterations = j.value("iterations", 256);
    p.bailout   = j.value("bailout",   2.0);
    const double iso = j.value("iso",  0.5);

    if (p.resolution < 8 || p.resolution > 1024) throw std::runtime_error("invalid resolution");
    if (p.iterations < 1 || p.iterations > 10000) throw std::runtime_error("invalid iterations");

    auto run = runner.createRun("transition-mesh", body);
    runner.setStatus(run.id, "running");

    double fieldMs = 0.0, mcMs = 0.0;
    size_t vc = 0, tc = 0;

    try {
        const auto t0 = std::chrono::steady_clock::now();
        compute::McField field = compute::buildTransitionVolume(p);
        const auto t1 = std::chrono::steady_clock::now();
        compute::Mesh mesh = compute::marchingCubes(field, static_cast<float>(iso));
        const auto t2 = std::chrono::steady_clock::now();
        fieldMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        mcMs    = std::chrono::duration<double, std::milli>(t2 - t1).count();
        vc = mesh.vertices.size();
        tc = mesh.triangleCount();

        if (vc == 0) throw std::runtime_error("empty mesh (iso value gives no surface)");

        const std::filesystem::path glbPath =
            std::filesystem::path(run.outputDir) / "transition_mesh.glb";
        const std::filesystem::path stlPath =
            std::filesystem::path(run.outputDir) / "transition_mesh.stl";
        compute::writeGlb(glbPath.string(), mesh);
        compute::writeStlBinary(stlPath.string(), mesh);

        runner.addArtifact(run.id, Artifact{"transition-mesh", glbPath.string(), "mesh"});
        runner.addArtifact(run.id, Artifact{"transition-mesh", stlPath.string(), "stl"});
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    const std::string glbId = run.id + ":transition_mesh.glb";
    const std::string stlId = run.id + ":transition_mesh.stl";
    Json resp = {
        {"runId",      run.id},
        {"status",     "completed"},
        {"glbArtifactId", glbId},
        {"stlArtifactId", stlId},
        {"glbUrl",     "/api/artifacts/content?artifactId=" + glbId},
        {"stlUrl",     "/api/artifacts/download?artifactId=" + stlId},
        {"vertexCount", vc},
        {"triangleCount", tc},
        {"fieldMs",  fieldMs},
        {"mcMs",     mcMs},
    };
    return resp.dump();
}

} // namespace fsd
