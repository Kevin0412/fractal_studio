// routes_ln.cpp
//
// ln-map strip renderer. Produces a single tall PNG whose pixel columns are
// angles θ ∈ [0, 2π) and pixel rows are log-radii descending from ln(4) (the
// outside of the set) toward ln(4) − rows·2π/s (arbitrarily deep). This is
// the same parameterisation C_mandelbrot/big_png_ln.py uses.
//
// For a given row `r` and column `x`:
//     θ = 2π · x / s
//     k = ln(4) − r · 2π / s
//     c = center + e^k · (cos θ + i sin θ)
//
// A single strip rendered once captures the entire zoom sequence around the
// target `center`. In Phase 2, routes_video.cpp will slide a window down the
// strip and exp-warp it into cartesian video frames — no fractal recompute.

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/variants.hpp"
#include "../compute/escape_time.hpp"
#include "../compute/colormap.hpp"
#include "../compute/image_io.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif
#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fsd {

namespace {

constexpr double TAU     = 6.283185307179586;
constexpr double PI      = 3.141592653589793;
constexpr double LN_TWO  = 0.6931471805599453;
constexpr double LN_FOUR = 1.3862943611198906;

int roundUpToMultiple(int value, int multiple) {
    if (multiple <= 1) return value;
    const int rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

int derivedMinStripWidth(int W, int H) {
    const double diag = std::sqrt(static_cast<double>(W) * static_cast<double>(W)
                                + static_cast<double>(H) * static_cast<double>(H));
    const int minWidth = static_cast<int>(std::ceil(diag * PI));
    return roundUpToMultiple(minWidth, 8);
}

template <compute::Variant V>
void render_ln_strip(
    double cr, double ci,
    int s, int t,
    int iters, double bailout, double bailoutSq,
    compute::Colormap colormap,
    cv::Mat& out
) {
    #pragma omp parallel
    {
        std::vector<compute::Cx<double>> orbit_scratch;
        #pragma omp for schedule(dynamic, 8)
        for (int row = 0; row < t; row++) {
            uint8_t* rowp = out.ptr<uint8_t>(row);
            const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(s);
            const double r_mag = std::exp(k);
            for (int x = 0; x < s; x++) {
                const double th = TAU * static_cast<double>(x) / static_cast<double>(s);
                const double ore = cr + r_mag * std::cos(th);
                const double oim = ci + r_mag * std::sin(th);
                const compute::Cx<double> c{ore, oim};
                const compute::Cx<double> z0{0.0, 0.0};
                const compute::IterResult ir = compute::iterate<V, double>(
                    z0, c, iters, bailout, bailoutSq, compute::Metric::Escape, 1, orbit_scratch);
                uint8_t* px = rowp + 3 * x;
                const int    it   = ir.escaped ? ir.iter : iters;
                const double norm = ir.escaped ? ir.norm : 0.0;
                compute::colorize_escape_bgr(it, iters, colormap, norm, false, px[0], px[1], px[2]);
            }
        }
    }
}

void dispatch_ln_strip(
    compute::Variant v,
    double cr, double ci,
    int s, int t,
    int iters, double bailout, double bailoutSq,
    compute::Colormap colormap,
    cv::Mat& out
) {
    using V = compute::Variant;
    switch (v) {
        case V::Mandelbrot: render_ln_strip<V::Mandelbrot>(cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Tri:        render_ln_strip<V::Tri>       (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Boat:       render_ln_strip<V::Boat>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Duck:       render_ln_strip<V::Duck>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Bell:       render_ln_strip<V::Bell>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Fish:       render_ln_strip<V::Fish>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Vase:       render_ln_strip<V::Vase>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Bird:       render_ln_strip<V::Bird>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Mask:       render_ln_strip<V::Mask>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::Ship:       render_ln_strip<V::Ship>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::SinZ:       render_ln_strip<V::SinZ>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::CosZ:       render_ln_strip<V::CosZ>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::ExpZ:       render_ln_strip<V::ExpZ>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::SinhZ:      render_ln_strip<V::SinhZ>     (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::CoshZ:      render_ln_strip<V::CoshZ>     (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
        case V::TanZ:       render_ln_strip<V::TanZ>      (cr, ci, s, t, iters, bailout, bailoutSq, colormap, out); break;
    }
}

} // namespace

std::string lnMapRenderRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    (void)repoRoot;
    const Json j = parseJsonBody(body);

    const double cr            = j.value("centerRe", 0.0);
    const double ci            = j.value("centerIm", 0.0);
    const int    outW          = j.value("width", 0);
    const int    outH          = j.value("height", 0);
    int s = j.value("widthS", 1920);
    if (outW > 0 && outH > 0) {
        s = std::max(s, derivedMinStripWidth(outW, outH));
        s = roundUpToMultiple(s, 8);
    }
    const double depthOctaves  = j.value("depthOctaves", 40.0);
    const std::string variantStr  = j.value("variant",  std::string("mandelbrot"));
    const std::string colormapStr = j.value("colorMap", std::string("classic_cos"));
    const int iters            = j.value("iterations", 4096);

    if (s < 128 || s > 8192)               throw std::runtime_error("invalid widthS (128..8192)");
    if (depthOctaves < 1.0 || depthOctaves > 80.0) throw std::runtime_error("invalid depthOctaves (1..80)");
    if (iters < 1 || iters > 10000000)     throw std::runtime_error("invalid iterations");

    // Same formula as big_png_ln.py:20 — 2 base octaves + requested depth.
    const double t_exact = (2.0 + depthOctaves) * LN_TWO / TAU * static_cast<double>(s);
    const int t = static_cast<int>(std::ceil(t_exact));

    compute::Variant v;
    if (!compute::variant_from_name(variantStr.c_str(), v)) v = compute::Variant::Mandelbrot;
    double bailout = j.contains("bailout") && !j["bailout"].is_null()
        ? j.value("bailout", 2.0)
        : compute::variant_default_bailout(v);
    const double bailoutSq = j.contains("bailoutSq") && !j["bailoutSq"].is_null()
        ? j.value("bailoutSq", compute::variant_default_bailout_sq(v))
        : (j.contains("bailout") && !j["bailout"].is_null()
            ? bailout * bailout
            : compute::variant_default_bailout_sq(v));
    if (j.contains("bailoutSq") && !j["bailoutSq"].is_null() &&
        !(j.contains("bailout") && !j["bailout"].is_null())) {
        bailout = std::sqrt(bailoutSq);
    }
    if (!(bailout > 0.0) || !std::isfinite(bailout)) throw std::runtime_error("invalid bailout");
    if (!(bailoutSq > 0.0) || !std::isfinite(bailoutSq)) throw std::runtime_error("invalid bailoutSq");
    compute::Colormap cm;
    if (!compute::colormap_from_name(colormapStr.c_str(), cm)) cm = compute::Colormap::ClassicCos;

    auto run = runner.createRun("ln-map", body);
    runner.setStatus(run.id, "running");

    std::string pngPath;
    double elapsed = 0.0;

    try {
        cv::Mat strip(t, s, CV_8UC3);
        const auto t0 = std::chrono::steady_clock::now();
        dispatch_ln_strip(v, cr, ci, s, t, iters, bailout, bailoutSq, cm, strip);
        const auto t1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const std::filesystem::path stripPath =
            std::filesystem::path(run.outputDir) / "ln_map.png";
        compute::write_png(stripPath.string(), strip);
        pngPath = stripPath.string();
        runner.addArtifact(run.id, Artifact{"ln-map", pngPath, "image"});

        // Sidecar JSON so Phase 2 video export can read the parameters.
        Json sidecar = {
            {"centerRe",     cr},
            {"centerIm",     ci},
            {"widthS",       s},
            {"heightT",      t},
            {"depthOctaves", depthOctaves},
            {"lnRadiusTop",  LN_FOUR},
            {"variant",      variantStr},
            {"colorMap",     colormapStr},
            {"iterations",   iters},
            {"bailout",      bailout},
            {"bailoutSq",    bailoutSq},
        };
        const std::filesystem::path sidecarPath =
            std::filesystem::path(run.outputDir) / "ln_map.json";
        std::filesystem::create_directories(sidecarPath.parent_path());
        {
            std::ofstream os(sidecarPath);
            os << sidecar.dump(2);
        }
        runner.addArtifact(run.id, Artifact{"ln-map", sidecarPath.string(), "report"});
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    const std::string artifactId = run.id + ":ln_map.png";
    Json resp = {
        {"runId",       run.id},
        {"status",      "completed"},
        {"artifactId",  artifactId},
        {"imagePath",   "/api/artifacts/content?artifactId=" + artifactId},
        {"widthS",      s},
        {"heightT",     t},
        {"depthOctaves", depthOctaves},
        {"generatedMs", elapsed},
    };
    return resp.dump();
}

} // namespace fsd
