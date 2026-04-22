// routes_video.cpp
//
// Zoom video export from a ln-map artifact + a final rendered cartesian frame.
//
// Design
// ------
// The ln-map strip samples (angle θ, log-radius k) space around `center`.
// A video frame at zoom level `kTop` maps screen pixel (u, v) to:
//
//   world point c = center + (u + iv)·e^kTop          (u,v normalised)
//   strip col     = (θ / 2π) · s                      (wraps)
//   strip row     = (ln4 − kTop − ln(r_screen)) · s/τ
//
// where r_screen = |u+iv|. The corner pixel (r_screen = r_max = √(aspect²+1))
// hits strip row 0 when kTop_start = ln4 − ln(r_max). Starting there ensures
// no negative-row out-of-bounds at the first frame.
//
// The screen centre (r_screen→0) would require row→∞ — beyond the strip.
// That region is filled from a cartesian final image rendered directly at
// kTop_end = kTop_start − depth·ln2.
//
// Compositing rule (per pixel, per frame):
//   strip_row ∈ [0, stripH)  →  use bilinear sample from strip
//   else                      →  use bilinear sample from final image
//                                scaled by e^(kTop − kTop_end)

#include "routes.hpp"
#include "routes_common.hpp"

#include "../compute/image_io.hpp"
#include "../compute/map_kernel.hpp"
#include "../compute/variants.hpp"
#include "../compute/colormap.hpp"
#include "../compute/escape_time.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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

int resolveStripWidth(const Json& j, int W, int H) {
    const int derived = derivedMinStripWidth(W, H);
    const int requested = j.value("widthS", derived);
    return roundUpToMultiple(std::max(requested, derived), 8);
}

// ─── Julia-aware ln-strip renderer ───────────────────────────────────────────
// When julia=true, z₀ = center + e^k·e^(iθ) and c = fixed (juliaRe+juliaIm·i).
// When julia=false, c = center + e^k·e^(iθ) and z₀ = 0 (standard ln-map).

template <compute::Variant V>
void render_ln_strip_dispatch(
    bool julia,
    double cr, double ci,
    double jre, double jim,
    int s, int t,
    int iters, double bailout,
    compute::Colormap colormap,
    cv::Mat& out
) {
    const double bail2 = bailout * bailout;
    const compute::Cx<double> c_julia{jre, jim};

    #pragma omp parallel
    {
        std::vector<compute::Cx<double>> orbit_scratch;
        #pragma omp for schedule(dynamic, 8)
        for (int row = 0; row < t; row++) {
            uint8_t* rowp = out.ptr<uint8_t>(row);
            const double k     = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(s);
            const double r_mag = std::exp(k);
            for (int x = 0; x < s; x++) {
                const double th  = TAU * static_cast<double>(x) / static_cast<double>(s);
                const double pre = cr + r_mag * std::cos(th);
                const double pim = ci + r_mag * std::sin(th);
                compute::Cx<double> z0, c;
                if (julia) {
                    z0 = {pre, pim};
                    c  = c_julia;
                } else {
                    z0 = {0.0, 0.0};
                    c  = {pre, pim};
                }
                const compute::IterResult ir = compute::iterate<V, double>(
                    z0, c, iters, bail2, compute::Metric::Escape, 1, orbit_scratch);
                uint8_t* px = rowp + 3 * x;
                const int    it   = ir.escaped ? ir.iter : iters;
                const double norm = ir.escaped ? ir.norm : 0.0;
                compute::colorize_escape_bgr(it, iters, colormap, norm, false, px[0], px[1], px[2]);
            }
        }
    }
}

void dispatch_ln_strip_full(
    compute::Variant v,
    bool julia,
    double cr, double ci,
    double jre, double jim,
    int s, int t,
    int iters, double bailout,
    compute::Colormap colormap,
    cv::Mat& out
) {
    using V = compute::Variant;
#define CASE(X) case V::X: render_ln_strip_dispatch<V::X>(julia,cr,ci,jre,jim,s,t,iters,bailout,colormap,out); break
    switch (v) {
        CASE(Mandelbrot); CASE(Tri); CASE(Boat); CASE(Duck); CASE(Bell);
        CASE(Fish);       CASE(Vase); CASE(Bird); CASE(Mask); CASE(Ship);
        CASE(SinZ);       CASE(CosZ); CASE(ExpZ); CASE(SinhZ); CASE(CoshZ); CASE(TanZ);
    }
#undef CASE
}

void renderWarpFrameShared(
    const cv::Mat& stripWrap,
    const cv::Mat& finalImg,
    int W, int H,
    double kTop, double kTop_end,
    cv::Mat& frame,
    cv::Mat& stripFrame,
    cv::Mat& finalFrame,
    cv::Mat& mapX,
    cv::Mat& mapY,
    cv::Mat& fmapX,
    cv::Mat& fmapY,
    std::vector<float>& useStrip
) {
    const double aspect = static_cast<double>(W) / static_cast<double>(H);
    const int stripH    = stripWrap.rows;
    const int s         = stripWrap.cols - 1;
    const double S      = std::exp(kTop - kTop_end);

    for (int y = 0; y < H; ++y) {
        const double vy = -(2.0 * (y + 0.5) / H - 1.0);
        for (int x = 0; x < W; ++x) {
            const double ux = (2.0 * (x + 0.5) / W - 1.0) * aspect;
            const double r2 = ux * ux + vy * vy;
            double th = std::atan2(vy, ux);
            if (th < 0.0) th += TAU;

            double row = -1.0;
            if (r2 > 1e-30) {
                const double lnR = 0.5 * std::log(r2);
                row = (LN_FOUR - kTop - lnR) * s / TAU;
            }
            const double col = th / TAU * s;
            const size_t idx = static_cast<size_t>(y) * W + x;

            if (row >= 0.0 && row < static_cast<double>(stripH) - 1.0) {
                mapX.at<float>(y, x) = static_cast<float>(col);
                mapY.at<float>(y, x) = static_cast<float>(row);
                useStrip[idx] = 1.0f;
            } else {
                mapX.at<float>(y, x) = -1.0f;
                mapY.at<float>(y, x) = -1.0f;
                useStrip[idx] = 0.0f;
            }

            const double fu = ux * S;
            const double fv = vy * S;
            fmapX.at<float>(y, x) = static_cast<float>((fu / aspect * 0.5 + 0.5) * W);
            fmapY.at<float>(y, x) = static_cast<float>((-fv * 0.5 + 0.5) * H);
        }
    }

    cv::remap(stripWrap, stripFrame, mapX,  mapY,  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    cv::remap(finalImg,  finalFrame, fmapX, fmapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    for (int y = 0; y < H; ++y) {
        const uint8_t* sp = stripFrame.ptr<uint8_t>(y);
        const uint8_t* fp = finalFrame.ptr<uint8_t>(y);
        uint8_t* dp = frame.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const size_t idx = static_cast<size_t>(y) * W + x;
            if (useStrip[idx] > 0.5f) {
                dp[3*x+0] = sp[3*x+0];
                dp[3*x+1] = sp[3*x+1];
                dp[3*x+2] = sp[3*x+2];
            } else {
                dp[3*x+0] = fp[3*x+0];
                dp[3*x+1] = fp[3*x+1];
                dp[3*x+2] = fp[3*x+2];
            }
        }
    }
}

// ─── Shared video generation helper ──────────────────────────────────────────
static std::string generateZoomVideo(
    const cv::Mat& strip,
    const cv::Mat& finalImg,
    int W, int H, int fps, int frameCount,
    double kTop_start, double kTop_end, double depthOctaves,
    const std::filesystem::path& outDir, const std::string& baseName
) {
    cv::Mat stripWrap;
    cv::copyMakeBorder(strip, stripWrap, 0, 0, 0, 1, cv::BORDER_WRAP);

    const std::filesystem::path mp4 = outDir / (baseName + ".mp4");
    const std::filesystem::path avi = outDir / (baseName + ".avi");

    cv::Mat frame(H, W, CV_8UC3);
    cv::Mat stripFrame(H, W, CV_8UC3);
    cv::Mat finalFrame(H, W, CV_8UC3);
    cv::Mat mapX(H, W, CV_32FC1);
    cv::Mat mapY(H, W, CV_32FC1);
    cv::Mat fmapX(H, W, CV_32FC1);
    cv::Mat fmapY(H, W, CV_32FC1);
    std::vector<float> useStrip(static_cast<size_t>(W) * H, 0.0f);

    auto renderFrames = [&](auto&& writeFrame) {
        for (int f = 0; f < frameCount; f++) {
            const double tNorm = static_cast<double>(f) / std::max(1, frameCount - 1);
            const double kTop  = kTop_start - tNorm * depthOctaves * LN_TWO;
            renderWarpFrameShared(stripWrap, finalImg, W, H, kTop, kTop_end, frame, stripFrame, finalFrame, mapX, mapY, fmapX, fmapY, useStrip);
            writeFrame(frame);
        }
    };

    const std::string ffmpegCmd =
        "ffmpeg -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(W) + "x" + std::to_string(H) +
        " -r " + std::to_string(fps) +
        " -i - -an -c:v libx264rgb -preset medium -crf 0 \"" + mp4.string() + "\"";

    if (FILE* pipe = popen(ffmpegCmd.c_str(), "w")) {
        bool ok = true;
        try {
            renderFrames([&](const cv::Mat& rendered) {
                const size_t bytes = static_cast<size_t>(rendered.rows) * rendered.step;
                if (std::fwrite(rendered.data, 1, bytes, pipe) != bytes) {
                    ok = false;
                    throw std::runtime_error("ffmpeg pipe write failed");
                }
            });
        } catch (...) {
            pclose(pipe);
            throw;
        }
        const int rc = pclose(pipe);
        if (ok && rc == 0 && std::filesystem::exists(mp4)) return mp4.string();
    }

    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(mp4.string(), fourcc, static_cast<double>(fps), cv::Size(W, H), true);
    if (!writer.isOpened()) {
        writer.open(avi.string(), cv::VideoWriter::fourcc('M','J','P','G'), static_cast<double>(fps), cv::Size(W, H), true);
        if (!writer.isOpened()) throw std::runtime_error("VideoWriter failed");
    }

    renderFrames([&](const cv::Mat& rendered) {
        writer.write(rendered);
    });
    writer.release();

    if (std::filesystem::exists(mp4)) return mp4.string();
    if (std::filesystem::exists(avi)) return avi.string();
    return mp4.string();
}

struct LnMapLookup {
    std::filesystem::path pngPath;
    Json sidecar;
};

LnMapLookup resolveLnMap(const std::filesystem::path& repoRoot, const std::string& artifactId) {
    namespace fs = std::filesystem;
    const auto split = artifactId.find(':');
    if (split == std::string::npos) throw std::runtime_error("bad artifactId");
    const std::string runId   = artifactId.substr(0, split);
    const std::string fileName = artifactId.substr(split + 1);

    const fs::path runDir = repoRoot / "fractal_studio" / "runtime" / "runs" / runId;
    const fs::path png    = runDir / fileName;
    if (!fs::exists(png)) throw std::runtime_error("ln-map png not found: " + png.string());

    fs::path sidecar = runDir / "ln_map.json";
    if (!fs::exists(sidecar)) throw std::runtime_error("ln-map sidecar not found");

    std::ifstream in(sidecar);
    std::ostringstream ss; ss << in.rdbuf();
    return { png, Json::parse(ss.str()) };
}

} // namespace

std::string zoomVideoRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    const Json j = parseJsonBody(body);
    const std::string lnArtifactId = j.value("lnMapArtifactId", std::string(""));
    if (lnArtifactId.empty()) throw std::runtime_error("lnMapArtifactId required");

    const int    fps         = j.value("fps",          30);
    const double durationSec = j.value("durationSec",  8.0);
    const int    W           = j.value("width",        720);
    const int    H           = j.value("height",       720);

    if (fps < 1 || fps > 120)                  throw std::runtime_error("invalid fps (1..120)");
    if (durationSec <= 0 || durationSec > 300) throw std::runtime_error("invalid durationSec (0..300)");
    if (W < 128 || W > 1920 || H < 128 || H > 1080) throw std::runtime_error("invalid output size");

    LnMapLookup lk = resolveLnMap(repoRoot, lnArtifactId);
    cv::Mat strip  = compute::read_png(lk.pngPath.string());

    const int    stripW       = strip.cols;
    const int    stripH       = strip.rows;
    const int    s            = lk.sidecar.value("widthS",      stripW);
    const double sidecarDepth = lk.sidecar.value("depthOctaves", 40.0);
    const double cr           = lk.sidecar.value("centerRe",    0.0);
    const double ci           = lk.sidecar.value("centerIm",    0.0);
    const int    iters        = lk.sidecar.value("iterations",  4096);
    const std::string variantStr  = lk.sidecar.value("variant",  std::string("mandelbrot"));
    const std::string colormapStr = lk.sidecar.value("colorMap", std::string("classic_cos"));

    double depthOctaves = j.value("depthOctaves", 0.0);
    if (depthOctaves <= 0.0) depthOctaves = sidecarDepth - 1.5;

    // ── startKTop: corner pixel of first frame hits strip row 0 ──────────────
    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double r_max   = std::sqrt(aspect * aspect + 1.0);  // half-diagonal in normalised coords
    const double kTop_start = LN_FOUR - std::log(r_max);
    const double kTop_end   = kTop_start - depthOctaves * LN_TWO;

    // ── Render final cartesian image at kTop_end ──────────────────────────────
    // This fills the centre pixels that the ln-map strip can't reach.
    compute::Variant variantVal;
    if (!compute::variant_from_name(variantStr.c_str(), variantVal))
        variantVal = compute::Variant::Mandelbrot;
    compute::Colormap cmVal;
    if (!compute::colormap_from_name(colormapStr.c_str(), cmVal))
        cmVal = compute::Colormap::ClassicCos;

    // Scale: height of the cartesian view at kTop_end in complex units.
    // A pixel at normalised v=1 is at world height e^kTop_end, so full height = 2·e^kTop_end.
    compute::MapParams mp;
    mp.center_re  = cr;
    mp.center_im  = ci;
    mp.scale      = 2.0 * std::exp(kTop_end);
    mp.width      = W;
    mp.height     = H;
    mp.iterations = iters;
    mp.bailout    = 2.0;
    mp.variant    = variantVal;
    mp.metric     = compute::Metric::Escape;
    mp.colormap   = cmVal;
    mp.smooth     = false;
    mp.engine     = "auto";
    mp.scalar_type = "auto";

    cv::Mat finalImg(H, W, CV_8UC3);
    compute::render_map(mp, finalImg);

    // ── Video writer ──────────────────────────────────────────────────────────
    auto run = runner.createRun("zoom-video", body);
    runner.setStatus(run.id, "running");

    std::string mp4Path;
    double elapsed = 0.0;
    const int frameCount = static_cast<int>(std::round(durationSec * fps));

    try {
        const auto t0 = std::chrono::steady_clock::now();

        const std::filesystem::path outPath =
            std::filesystem::path(run.outputDir) / "zoom.mp4";
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(outPath.string(), fourcc, static_cast<double>(fps),
                               cv::Size(W, H), true);
        if (!writer.isOpened()) {
            const std::filesystem::path aviPath =
                std::filesystem::path(run.outputDir) / "zoom.avi";
            writer.open(aviPath.string(),
                        cv::VideoWriter::fourcc('M','J','P','G'),
                        static_cast<double>(fps), cv::Size(W, H), true);
            if (!writer.isOpened()) throw std::runtime_error("VideoWriter failed");
            mp4Path = aviPath.string();
        } else {
            mp4Path = outPath.string();
        }

        cv::Mat stripWrap;
        cv::copyMakeBorder(strip, stripWrap, 0, 0, 0, 1, cv::BORDER_WRAP);
        cv::Mat frame(H, W, CV_8UC3);
        cv::Mat stripFrame(H, W, CV_8UC3);
        cv::Mat finalFrame(H, W, CV_8UC3);
        cv::Mat mapX(H, W, CV_32FC1);
        cv::Mat mapY(H, W, CV_32FC1);
        cv::Mat fmapX(H, W, CV_32FC1);
        cv::Mat fmapY(H, W, CV_32FC1);
        std::vector<float> useStrip(static_cast<size_t>(W) * H, 0.0f);
        for (int f = 0; f < frameCount; f++) {
            const double t = static_cast<double>(f) / std::max(1, frameCount - 1);
            const double kTop = kTop_start - t * depthOctaves * LN_TWO;
            renderWarpFrameShared(stripWrap, finalImg, W, H, kTop, kTop_end, frame, stripFrame, finalFrame, mapX, mapY, fmapX, fmapY, useStrip);
            writer.write(frame);
        }

        writer.release();
        const auto t1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        runner.addArtifact(run.id, Artifact{"zoom-video", mp4Path, "video"});
        runner.setStatus(run.id, "completed");
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }

    const std::string fname = std::filesystem::path(mp4Path).filename().string();
    const std::string artifactId = run.id + ":" + fname;
    Json resp = {
        {"runId",       run.id},
        {"status",      "completed"},
        {"artifactId",  artifactId},
        {"videoUrl",    "/api/artifacts/content?artifactId=" + artifactId},
        {"downloadUrl", "/api/artifacts/download?artifactId=" + artifactId},
        {"frameCount",  frameCount},
        {"fps",         fps},
        {"durationSec", durationSec},
        {"width",       W},
        {"height",      H},
        {"kTopStart",   kTop_start},
        {"kTopEnd",     kTop_end},
        {"depthOctaves",depthOctaves},
        {"generatedMs", elapsed},
    };
    return resp.dump();
}

// ─── Unified export: ln-map + final frame + video in one request ─────────────
//
// POST /api/video/export
// {
//   "centerRe", "centerIm", "julia", "juliaRe", "juliaIm",
//   "variant", "colorMap", "iterations",
//   "widthS", "depthOctaves",      // ln-map strip size
//   "fps", "durationSec", "width", "height"  // video output
// }
//
// Returns {videoArtifactId, lnMapArtifactId, finalFrameArtifactId, ...}.
// When julia=true, the ln-map samples z₀ around center with c=juliaRe+juliaIm·i.

std::string videoExportRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    (void)repoRoot;
    const Json j = parseJsonBody(body);

    const double cr         = j.value("centerRe", 0.0);
    const double ci         = j.value("centerIm", 0.0);
    const bool   julia      = j.value("julia",    false);
    const double jre        = j.value("juliaRe",  0.0);
    const double jim        = j.value("juliaIm",  0.0);
    const std::string variantStr  = j.value("variant",  std::string("mandelbrot"));
    const std::string colormapStr = j.value("colorMap", std::string("classic_cos"));
    const int    iters      = j.value("iterations", 2048);
    const double bailout    = j.value("bailout",   2.0);
    const double depth      = j.value("depthOctaves", 20.0);
    const int    fps        = j.value("fps",       30);
    const double durSec     = j.value("durationSec", 8.0);
    const int    W          = j.value("width",     720);
    const int    H          = j.value("height",    720);
    const int    s          = resolveStripWidth(j, W, H);

    if (s < 128 || s > 16384)               throw std::runtime_error("invalid widthS (128..16384)");
    if (depth < 1.0 || depth > 80.0)        throw std::runtime_error("invalid depthOctaves (1..80)");
    if (fps < 1 || fps > 120)               throw std::runtime_error("invalid fps (1..120)");
    if (durSec <= 0 || durSec > 300)        throw std::runtime_error("invalid durationSec (0..300)");
    if (W < 128 || W > 1920 || H < 128 || H > 1080) throw std::runtime_error("invalid output size");
    if (iters < 1 || iters > 10000000)      throw std::runtime_error("invalid iterations");

    compute::Variant v;
    if (!compute::variant_from_name(variantStr.c_str(), v)) v = compute::Variant::Mandelbrot;
    compute::Colormap cm;
    if (!compute::colormap_from_name(colormapStr.c_str(), cm)) cm = compute::Colormap::ClassicCos;

    auto run = runner.createRun("video-export", body);
    runner.setStatus(run.id, "running");

    try {
        const auto t0 = std::chrono::steady_clock::now();

        // ── 1. Render ln-map strip ─────────────────────────────────────────────
        const double t_exact = (2.0 + depth) * LN_TWO / TAU * static_cast<double>(s);
        const int t = static_cast<int>(std::ceil(t_exact));
        cv::Mat strip(t, s, CV_8UC3);
        dispatch_ln_strip_full(v, julia, cr, ci, jre, jim, s, t, iters, bailout, cm, strip);

        const std::filesystem::path stripPath = std::filesystem::path(run.outputDir) / "ln_map.png";
        compute::write_png(stripPath.string(), strip);
        runner.addArtifact(run.id, Artifact{"ln-map", stripPath.string(), "image"});

        // Sidecar so old zoomVideoRoute can also consume this ln-map.
        {
            Json sc = {
                {"centerRe", cr}, {"centerIm", ci},
                {"julia", julia}, {"juliaRe", jre}, {"juliaIm", jim},
                {"widthS", s}, {"heightT", t}, {"depthOctaves", depth},
                {"lnRadiusTop", LN_FOUR}, {"variant", variantStr},
                {"colorMap", colormapStr}, {"iterations", iters},
            };
            const std::filesystem::path scPath = std::filesystem::path(run.outputDir) / "ln_map.json";
            std::ofstream os(scPath);
            os << sc.dump(2);
        }

        // ── 2. Render final cartesian frame ────────────────────────────────────
        const double aspect     = static_cast<double>(W) / static_cast<double>(H);
        const double r_max      = std::sqrt(aspect * aspect + 1.0);
        const double kTop_start = LN_FOUR - std::log(r_max);
        const double kTop_end   = kTop_start - depth * LN_TWO;

        compute::MapParams mp;
        mp.center_re  = cr;
        mp.center_im  = ci;
        mp.scale      = 2.0 * std::exp(kTop_end);
        mp.width      = W;
        mp.height     = H;
        mp.iterations = iters;
        mp.bailout    = bailout;
        mp.variant    = v;
        mp.metric     = compute::Metric::Escape;
        mp.colormap   = cm;
        mp.smooth     = false;
        mp.julia      = julia;
        mp.julia_re   = jre;
        mp.julia_im   = jim;
        mp.engine     = "auto";
        mp.scalar_type = "auto";

        cv::Mat finalImg(H, W, CV_8UC3);
        compute::render_map(mp, finalImg);

        const std::filesystem::path finalPath = std::filesystem::path(run.outputDir) / "final_frame.png";
        compute::write_png(finalPath.string(), finalImg);
        runner.addArtifact(run.id, Artifact{"final-frame", finalPath.string(), "image"});

        // ── 3. Generate video ──────────────────────────────────────────────────
        const int frameCount = static_cast<int>(std::round(durSec * fps));
        const std::string videoPath = generateZoomVideo(
            strip, finalImg, W, H, fps, frameCount,
            kTop_start, kTop_end, depth,
            std::filesystem::path(run.outputDir), "zoom"
        );
        const std::string videoFile = std::filesystem::path(videoPath).filename().string();
        runner.addArtifact(run.id, Artifact{"zoom-video", videoPath, "video"});

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        runner.setStatus(run.id, "completed");

        const std::string lnArtId    = run.id + ":ln_map.png";
        const std::string finalArtId = run.id + ":final_frame.png";
        const std::string videoArtId = run.id + ":" + videoFile;

        Json resp = {
            {"runId",              run.id},
            {"status",             "completed"},
            {"videoArtifactId",    videoArtId},
            {"videoUrl",           "/api/artifacts/content?artifactId="  + videoArtId},
            {"videoDownloadUrl",   "/api/artifacts/download?artifactId=" + videoArtId},
            {"lnMapArtifactId",    lnArtId},
            {"lnMapDownloadUrl",   "/api/artifacts/download?artifactId=" + lnArtId},
            {"finalFrameArtifactId", finalArtId},
            {"finalFrameDownloadUrl", "/api/artifacts/download?artifactId=" + finalArtId},
            {"frameCount",         frameCount},
            {"fps",                fps},
            {"durationSec",        durSec},
            {"width",              W},
            {"height",             H},
            {"generatedMs",        elapsed},
        };
        return resp.dump();

    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }
}

} // namespace fsd
