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
#include "../compute/ln_map.hpp"
#include "../compute/map_kernel.hpp"
#include "../compute/variants.hpp"
#include "../compute/colormap.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "../compute/cuda/video_warp.cuh"
#  define USE_CUDA_VIDEO_WARP 1
#else
#  define USE_CUDA_VIDEO_WARP 0
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fsd {

namespace {

constexpr double TAU     = 6.283185307179586;
constexpr double PI      = 3.141592653589793;
constexpr double LN_TWO  = 0.6931471805599453;
constexpr double LN_FOUR = 1.3862943611198906;
constexpr int    MIN_VIDEO_DIM = 128;
constexpr int    MAX_VIDEO_DIM = 8192;
constexpr int64_t MAX_VIDEO_PIXELS = 7680LL * 4320LL;
constexpr double DEFAULT_SECONDS_PER_OCTAVE = 0.4;
constexpr double MAX_SECONDS_PER_OCTAVE = 60.0;
constexpr double MAX_VIDEO_DURATION_SEC = 3600.0;

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

struct StripPlan {
    int fullWidthS = 0;
    int actualWidthS = 0;
    int heightT = 0;
    double qualityScale = 1.0;
    std::string qualityPreset = "full";
    uint64_t estimatedPeakMemory = 0;
};

double presetQualityScale(const std::string& preset) {
    if (preset == "draft") return 0.35;
    if (preset == "balanced") return 0.55;
    if (preset == "high") return 0.75;
    if (preset == "full") return 1.0;
    return 0.55;
}

std::string defaultQualityPresetForSize(int W, int H) {
    return (W >= 3840 || H >= 2160) ? "balanced" : "high";
}

uint64_t estimateVideoPeakMemoryBytes(int W, int H, int s, int t) {
    const uint64_t pixels = static_cast<uint64_t>(W) * static_cast<uint64_t>(H);
    const uint64_t stripPixels = static_cast<uint64_t>(s) * static_cast<uint64_t>(t);
    const uint64_t frameBgr = pixels * 3u;
    const uint64_t remapF32 = pixels * 4u;
    const uint64_t stripBgr = stripPixels * 3u;
    return stripBgr + frameBgr * 4u + remapF32 * 5u + pixels;
}

bool commandSucceeds(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

StripPlan resolveStripPlan(const Json& j, int W, int H, double depthOctaves) {
    StripPlan plan;
    plan.fullWidthS = derivedMinStripWidth(W, H);
    plan.qualityPreset = j.value("qualityPreset", defaultQualityPresetForSize(W, H));
    plan.qualityScale = j.value("qualityScale", presetQualityScale(plan.qualityPreset));
    if (!(plan.qualityScale > 0.0) || plan.qualityScale > 1.0 || !std::isfinite(plan.qualityScale)) {
        throw std::runtime_error("invalid qualityScale (0..1)");
    }

    int requested = 0;
    if (j.contains("widthS") && !j["widthS"].is_null()) {
        requested = j.value("widthS", plan.fullWidthS);
        plan.qualityPreset = "custom";
        plan.qualityScale = static_cast<double>(requested) / std::max(1, plan.fullWidthS);
    } else {
        requested = static_cast<int>(std::ceil(static_cast<double>(plan.fullWidthS) * plan.qualityScale));
    }
    plan.actualWidthS = roundUpToMultiple(std::max(128, requested), 8);
    const double t_exact = (2.0 + depthOctaves) * LN_TWO / TAU * static_cast<double>(plan.actualWidthS);
    plan.heightT = static_cast<int>(std::ceil(t_exact));
    plan.estimatedPeakMemory = estimateVideoPeakMemoryBytes(W, H, plan.actualWidthS, plan.heightT);
    return plan;
}

void validateVideoOutputSize(int W, int H) {
    if (W < MIN_VIDEO_DIM || H < MIN_VIDEO_DIM || W > MAX_VIDEO_DIM || H > MAX_VIDEO_DIM) {
        throw std::runtime_error("invalid output size (128..8192)");
    }
    const int64_t pixels = static_cast<int64_t>(W) * static_cast<int64_t>(H);
    if (pixels > MAX_VIDEO_PIXELS) {
        throw std::runtime_error("invalid output size (too many pixels; max 7680x4320 area)");
    }
}

std::pair<int, int> resolvePreviewSize(const Json& j, int W, int H) {
    int previewW = j.value("previewWidth", 0);
    int previewH = j.value("previewHeight", 0);
    if (previewW <= 0 || previewH <= 0) {
        constexpr double maxPreviewSide = 720.0;
        const double scale = std::min(1.0, maxPreviewSide / static_cast<double>(std::max(W, H)));
        previewW = std::max(64, static_cast<int>(std::llround(static_cast<double>(W) * scale)));
        previewH = std::max(64, static_cast<int>(std::llround(static_cast<double>(H) * scale)));
    }
    if (previewW < 64 || previewH < 64 || previewW > 2048 || previewH > 2048) {
        throw std::runtime_error("invalid preview size (64..2048)");
    }
    if (static_cast<int64_t>(previewW) * static_cast<int64_t>(previewH) > 1920LL * 1080LL) {
        throw std::runtime_error("invalid preview size (too many pixels)");
    }
    return { previewW, previewH };
}

double kTopStartForFrame(int W, int H) {
    const double aspect = static_cast<double>(W) / static_cast<double>(H);
    const double rMax   = std::sqrt(aspect * aspect + 1.0);
    return LN_FOUR - std::log(rMax);
}

double resolveDepthOctaves(const Json& j, double kTopStart, double fallbackDepth) {
    double depth = j.value("depthOctaves", fallbackDepth);
    if (j.contains("targetScale") && !j["targetScale"].is_null()) {
        const double targetScale = j.value("targetScale", 0.0);
        if (!(targetScale > 0.0) || !std::isfinite(targetScale)) {
            throw std::runtime_error("invalid targetScale");
        }
        const double targetKTop = std::log(targetScale * 0.5);
        depth = (kTopStart - targetKTop) / LN_TWO;
    }
    if (!(depth > 0.0) || !std::isfinite(depth)) {
        throw std::runtime_error("invalid depthOctaves");
    }
    return depth;
}

double resolveSecondsPerOctave(const Json& j, double depthOctaves) {
    double secondsPerOctave = DEFAULT_SECONDS_PER_OCTAVE;
    if (j.contains("secondsPerOctave") && !j["secondsPerOctave"].is_null()) {
        secondsPerOctave = j.value("secondsPerOctave", DEFAULT_SECONDS_PER_OCTAVE);
    } else if (j.contains("durationSec") && !j["durationSec"].is_null()) {
        const double durationSec = j.value("durationSec", 0.0);
        if (!(durationSec > 0.0) || !std::isfinite(durationSec)) {
            throw std::runtime_error("invalid durationSec");
        }
        secondsPerOctave = durationSec / std::max(depthOctaves, 1e-9);
    }
    if (!(secondsPerOctave > 0.0) || secondsPerOctave > MAX_SECONDS_PER_OCTAVE || !std::isfinite(secondsPerOctave)) {
        throw std::runtime_error("invalid secondsPerOctave (0..60)");
    }
    return secondsPerOctave;
}

int frameCountFromSpeed(double depthOctaves, double secondsPerOctave, int fps, double& durationSec) {
    durationSec = depthOctaves * secondsPerOctave;
    if (!(durationSec > 0.0) || durationSec > MAX_VIDEO_DURATION_SEC || !std::isfinite(durationSec)) {
        throw std::runtime_error("invalid video duration");
    }
    const long long frames = std::llround(durationSec * static_cast<double>(fps));
    if (frames < 2) return 2;
    if (frames > 10000000LL) throw std::runtime_error("too many video frames");
    return static_cast<int>(frames);
}

double bailoutSqFromJson(const Json& j, double radius, double defaultSq) {
    if (j.contains("bailoutSq") && !j["bailoutSq"].is_null()) {
        return j.value("bailoutSq", defaultSq);
    }
    if (j.contains("bailout") && !j["bailout"].is_null()) {
        return radius * radius;
    }
    return defaultSq;
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

cv::Mat renderZoomPreviewFrame(
    const cv::Mat& strip,
    const cv::Mat& finalImg,
    int W, int H,
    double kTop,
    double kTop_end
) {
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

    renderWarpFrameShared(stripWrap, finalImg, W, H, kTop, kTop_end,
                          frame, stripFrame, finalFrame, mapX, mapY, fmapX, fmapY, useStrip);
    return frame;
}

// ─── Shared video generation helper ──────────────────────────────────────────
static std::string generateZoomVideo(
    const cv::Mat& strip,
    const cv::Mat& finalImg,
    int W, int H, int fps, int frameCount,
    double kTop_start, double kTop_end, double depthOctaves,
    const std::filesystem::path& outDir, const std::string& baseName,
    const std::function<void(int)>& on_frame_done = nullptr,
    std::string* ffmpeg_stderr_out = nullptr,
    std::string* encoder_out = nullptr,
    bool prefer_cuda_warp = true,
    std::string* warp_method_out = nullptr
) {
    if (encoder_out) *encoder_out = "";
    if (warp_method_out) *warp_method_out = "opencv_cpu_remap";

    cv::Mat stripWrap;
    cv::copyMakeBorder(strip, stripWrap, 0, 0, 0, 1, cv::BORDER_WRAP);

#if USE_CUDA_VIDEO_WARP
    fsd_cuda::CudaVideoWarpContext cudaWarp;
    struct CudaWarpGuard {
        fsd_cuda::CudaVideoWarpContext* ctx = nullptr;
        ~CudaWarpGuard() { if (ctx) fsd_cuda::cuda_video_warp_release(*ctx); }
    } cudaWarpGuard{&cudaWarp};
    bool useCudaWarp = false;
    if (prefer_cuda_warp && fsd_cuda::cuda_video_warp_available()) {
        try {
            fsd_cuda::cuda_video_warp_init(stripWrap, finalImg, cudaWarp);
            useCudaWarp = true;
            if (warp_method_out) *warp_method_out = "cuda_kernel";
        } catch (...) {
            fsd_cuda::cuda_video_warp_release(cudaWarp);
            useCudaWarp = false;
        }
    }
#else
    (void)prefer_cuda_warp;
#endif

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
#if USE_CUDA_VIDEO_WARP
            if (useCudaWarp) {
                fsd_cuda::cuda_video_warp_frame(cudaWarp, kTop, kTop_end, frame);
            } else
#endif
            renderWarpFrameShared(stripWrap, finalImg, W, H, kTop, kTop_end, frame, stripFrame, finalFrame, mapX, mapY, fmapX, fmapY, useStrip);
            writeFrame(frame);
            if (on_frame_done && (f + 1 == frameCount || ((f + 1) % 8) == 0)) on_frame_done(f + 1);
        }
    };

    const std::filesystem::path ffmpegErr = outDir / (baseName + "_ffmpeg.stderr.txt");
    std::vector<std::pair<std::string, std::string>> ffmpegCmds;
    const std::string inputArgs =
        "ffmpeg -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(W) + "x" + std::to_string(H) +
        " -r " + std::to_string(fps) + " -i - -an ";
    if (commandSucceeds("bash -lc \"ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc\"")) {
        ffmpegCmds.push_back({"h264_nvenc",
            inputArgs + "-c:v h264_nvenc -pix_fmt yuv420p -preset p5 -cq 18 \"" + mp4.string() +
            "\" 2>\"" + ffmpegErr.string() + "\""});
    }
    if (commandSucceeds("bash -lc \"ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_nvenc\"")) {
        ffmpegCmds.push_back({"hevc_nvenc",
            inputArgs + "-c:v hevc_nvenc -pix_fmt yuv420p -preset p5 -cq 20 \"" + mp4.string() +
            "\" 2>\"" + ffmpegErr.string() + "\""});
    }
    ffmpegCmds.push_back({"libx264",
        inputArgs + "-c:v libx264 -pix_fmt yuv420p -preset medium -crf 16 \"" + mp4.string() +
        "\" 2>\"" + ffmpegErr.string() + "\""});

    for (const auto& [encoderName, ffmpegCmd] : ffmpegCmds) {
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
            if (ffmpeg_stderr_out) {
                std::ifstream errIn(ffmpegErr);
                std::ostringstream ss; ss << errIn.rdbuf();
                *ffmpeg_stderr_out = ss.str();
            }
            if (ok && rc == 0 && std::filesystem::exists(mp4)) {
                if (encoder_out) *encoder_out = encoderName;
                return mp4.string();
            }
        }
    }

    std::error_code removeEc;
    std::filesystem::remove(mp4, removeEc);
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(mp4.string(), fourcc, static_cast<double>(fps), cv::Size(W, H), true);
    std::string encoderName = "VideoWriter:mp4v";
    if (!writer.isOpened()) {
        writer.open(avi.string(), cv::VideoWriter::fourcc('M','J','P','G'), static_cast<double>(fps), cv::Size(W, H), true);
        encoderName = "VideoWriter:MJPG";
        if (!writer.isOpened()) {
            std::string msg = "VideoWriter failed";
            if (ffmpeg_stderr_out && !ffmpeg_stderr_out->empty()) {
                msg += "; ffmpeg stderr: " + *ffmpeg_stderr_out;
            }
            throw std::runtime_error(msg);
        }
    }

    renderFrames([&](const cv::Mat& rendered) {
        writer.write(rendered);
    });
    writer.release();

    if (std::filesystem::exists(mp4)) {
        if (encoder_out) *encoder_out = encoderName;
        return mp4.string();
    }
    if (std::filesystem::exists(avi)) {
        if (encoder_out) *encoder_out = encoderName;
        return avi.string();
    }
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

void setVideoProgress(
    JobRunner& runner,
    const std::string& runId,
    const std::string& stage,
    int current,
    int total,
    double depthCurrent,
    double depthTotal,
    const std::string& failedStage = "",
    const std::string& errorMessage = ""
) {
    Json j = {
        {"stage", stage},
        {"current", current},
        {"total", total},
        {"depthOctave", depthCurrent},
        {"totalDepthOctaves", depthTotal},
        {"failedStage", failedStage},
        {"errorMessage", errorMessage},
    };
    runner.setProgress(runId, j.dump());
}

} // namespace

std::string zoomVideoRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
    const Json j = parseJsonBody(body);
    const std::string lnArtifactId = j.value("lnMapArtifactId", std::string(""));
    if (lnArtifactId.empty()) throw std::runtime_error("lnMapArtifactId required");

    const int    fps         = j.value("fps",    30);
    const int    W           = j.value("width",  720);
    const int    H           = j.value("height", 720);

    if (fps < 1 || fps > 120) throw std::runtime_error("invalid fps (1..120)");
    validateVideoOutputSize(W, H);

    LnMapLookup lk = resolveLnMap(repoRoot, lnArtifactId);
    cv::Mat strip  = compute::read_png(lk.pngPath.string());

    const double sidecarDepth = lk.sidecar.value("depthOctaves", 40.0);
    const double cr           = lk.sidecar.value("centerRe",    0.0);
    const double ci           = lk.sidecar.value("centerIm",    0.0);
    const int    iters        = lk.sidecar.value("iterations",  4096);
    const std::string variantStr  = lk.sidecar.value("variant",  std::string("mandelbrot"));
    const std::string colormapStr = lk.sidecar.value("colorMap", std::string("classic_cos"));

    double depthOctaves = resolveDepthOctaves(j, kTopStartForFrame(W, H), sidecarDepth - 1.5);
    if (depthOctaves < 0.05 || depthOctaves > 120.0) {
        throw std::runtime_error("invalid depthOctaves (0.05..120)");
    }
    const double secondsPerOctave = resolveSecondsPerOctave(j, depthOctaves);
    double durationSec = 0.0;
    const int frameCount = frameCountFromSpeed(depthOctaves, secondsPerOctave, fps, durationSec);

    // ── startKTop: corner pixel of first frame hits strip row 0 ──────────────
    const double kTop_start = kTopStartForFrame(W, H);
    const double kTop_end   = kTop_start - depthOctaves * LN_TWO;

    // ── Render final cartesian image at kTop_end ──────────────────────────────
    // This fills the centre pixels that the ln-map strip can't reach.
    compute::Variant variantVal;
    if (!compute::variant_from_name(variantStr.c_str(), variantVal))
        variantVal = compute::Variant::Mandelbrot;
    double bailout = lk.sidecar.contains("bailout") && !lk.sidecar["bailout"].is_null()
        ? lk.sidecar.value("bailout", 2.0)
        : compute::variant_default_bailout(variantVal);
    const double bailoutSq = lk.sidecar.contains("bailoutSq") && !lk.sidecar["bailoutSq"].is_null()
        ? lk.sidecar.value("bailoutSq", compute::variant_default_bailout_sq(variantVal))
        : (lk.sidecar.contains("bailout") && !lk.sidecar["bailout"].is_null()
            ? bailout * bailout
            : compute::variant_default_bailout_sq(variantVal));
    if (lk.sidecar.contains("bailoutSq") && !lk.sidecar["bailoutSq"].is_null() &&
        !(lk.sidecar.contains("bailout") && !lk.sidecar["bailout"].is_null())) {
        bailout = std::sqrt(bailoutSq);
    }
    if (!(bailout > 0.0) || !std::isfinite(bailout)) throw std::runtime_error("invalid bailout");
    if (!(bailoutSq > 0.0) || !std::isfinite(bailoutSq)) throw std::runtime_error("invalid bailoutSq");
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
    mp.bailout    = bailout;
    mp.bailout_sq = bailoutSq;
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
    std::string ffmpegStderr;
    std::string encoderUsed;
    std::string warpMethod;
    double elapsed = 0.0;

    try {
        const auto t0 = std::chrono::steady_clock::now();
        setVideoProgress(runner, run.id, "video_warp_encode", 0, frameCount, depthOctaves, depthOctaves);
        mp4Path = generateZoomVideo(
            strip, finalImg, W, H, fps, frameCount,
            kTop_start, kTop_end, depthOctaves,
            std::filesystem::path(run.outputDir), "zoom",
            [&](int frameDone) {
                setVideoProgress(runner, run.id, "video_warp_encode", frameDone, frameCount, depthOctaves, depthOctaves);
            },
            &ffmpegStderr,
            &encoderUsed,
            j.value("cudaWarp", true),
            &warpMethod
        );
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
        {"secondsPerOctave", secondsPerOctave},
        {"width",       W},
        {"height",      H},
        {"kTopStart",   kTop_start},
        {"kTopEnd",     kTop_end},
        {"depthOctaves",depthOctaves},
        {"warpMethod",  warpMethod},
        {"encoder",     encoderUsed},
        {"ffmpegStderr",ffmpegStderr},
        {"generatedMs", elapsed},
    };
    return resp.dump();
}

// ─── Fast preview: direct-render start/end frames before video export ─────────
//
// This intentionally does not build the ln-map strip or encode video. It renders
// the first and final views at preview resolution, so the UI can tune depth
// before paying the full export cost.

std::string videoPreviewRoute(const std::filesystem::path& repoRoot, JobRunner& runner, const std::string& body) {
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
    const int    fps        = j.value("fps",       30);
    const int    W          = j.value("width",     720);
    const int    H          = j.value("height",    720);

    if (fps < 1 || fps > 120) throw std::runtime_error("invalid fps (1..120)");
    if (iters < 1 || iters > 10000000) throw std::runtime_error("invalid iterations");
    validateVideoOutputSize(W, H);
    const auto [previewW, previewH] = resolvePreviewSize(j, W, H);

    const double kTop_start = kTopStartForFrame(W, H);
    const double depth      = resolveDepthOctaves(j, kTop_start, 20.0);
    if (depth < 0.05 || depth > 120.0) throw std::runtime_error("invalid depthOctaves (0.05..120)");
    const double kTop_end   = kTop_start - depth * LN_TWO;
    const double secondsPerOctave = resolveSecondsPerOctave(j, depth);
    double durSec = 0.0;
    const int frameCount = frameCountFromSpeed(depth, secondsPerOctave, fps, durSec);

    compute::Variant v;
    if (!compute::variant_from_name(variantStr.c_str(), v)) v = compute::Variant::Mandelbrot;
    double bailout = j.contains("bailout") && !j["bailout"].is_null()
        ? j.value("bailout", 2.0)
        : compute::variant_default_bailout(v);
    const double bailoutSq = bailoutSqFromJson(j, bailout, compute::variant_default_bailout_sq(v));
    if (j.contains("bailoutSq") && !j["bailoutSq"].is_null() &&
        !(j.contains("bailout") && !j["bailout"].is_null())) {
        bailout = std::sqrt(bailoutSq);
    }
    if (!(bailout > 0.0) || !std::isfinite(bailout)) throw std::runtime_error("invalid bailout");
    if (!(bailoutSq > 0.0) || !std::isfinite(bailoutSq)) throw std::runtime_error("invalid bailoutSq");
    compute::Colormap cm;
    if (!compute::colormap_from_name(colormapStr.c_str(), cm)) cm = compute::Colormap::ClassicCos;

    auto run = runner.createRun("video-preview", body);
    runner.setStatus(run.id, "running");

    try {
        const auto t0 = std::chrono::steady_clock::now();

        auto renderPreview = [&](double kTop) {
            compute::MapParams mp;
            mp.center_re  = cr;
            mp.center_im  = ci;
            mp.scale      = 2.0 * std::exp(kTop);
            mp.width      = previewW;
            mp.height     = previewH;
            mp.iterations = iters;
            mp.bailout    = bailout;
            mp.bailout_sq = bailoutSq;
            mp.variant    = v;
            mp.metric     = compute::Metric::Escape;
            mp.colormap   = cm;
            mp.smooth     = false;
            mp.julia      = julia;
            mp.julia_re   = jre;
            mp.julia_im   = jim;
            mp.engine     = "auto";
            mp.scalar_type = "auto";

            cv::Mat img(previewH, previewW, CV_8UC3);
            compute::render_map(mp, img);
            return img;
        };

        const cv::Mat startPreview = renderPreview(kTop_start);
        const cv::Mat endPreview   = renderPreview(kTop_end);

        const std::filesystem::path startPath = std::filesystem::path(run.outputDir) / "start_frame.png";
        const std::filesystem::path endPath   = std::filesystem::path(run.outputDir) / "end_frame.png";
        compute::write_png(startPath.string(), startPreview);
        compute::write_png(endPath.string(), endPreview);
        runner.addArtifact(run.id, Artifact{"start-frame", startPath.string(), "image"});
        runner.addArtifact(run.id, Artifact{"end-frame",   endPath.string(),   "image"});

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        runner.setStatus(run.id, "completed");

        const std::string startArtId = run.id + ":start_frame.png";
        const std::string endArtId   = run.id + ":end_frame.png";
        Json resp = {
            {"runId",                run.id},
            {"status",               "completed"},
            {"startFrameArtifactId", startArtId},
            {"startFrameUrl",        "/api/artifacts/content?artifactId="  + startArtId},
            {"startFrameDownloadUrl","/api/artifacts/download?artifactId=" + startArtId},
            {"endFrameArtifactId",   endArtId},
            {"endFrameUrl",          "/api/artifacts/content?artifactId="  + endArtId},
            {"endFrameDownloadUrl",  "/api/artifacts/download?artifactId=" + endArtId},
            {"frameCount",           frameCount},
            {"fps",                  fps},
            {"durationSec",          durSec},
            {"secondsPerOctave",     secondsPerOctave},
            {"depthOctaves",         depth},
            {"targetScale",          2.0 * std::exp(kTop_end)},
            {"width",                previewW},
            {"height",               previewH},
            {"outputWidth",          W},
            {"outputHeight",         H},
            {"generatedMs",          elapsed},
        };
        return resp.dump();
    } catch (const std::exception&) {
        runner.setStatus(run.id, "failed");
        throw;
    }
}

// ─── Unified export: ln-map + final frame + video in one request ─────────────
//
// POST /api/video/export
// {
//   "centerRe", "centerIm", "julia", "juliaRe", "juliaIm",
//   "variant", "colorMap", "iterations",
//   "widthS", "depthOctaves",      // ln-map strip size
//   "fps", "secondsPerOctave", "width", "height"  // video output
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
    const int    fps        = j.value("fps",       30);
    const int    W          = j.value("width",     720);
    const int    H          = j.value("height",    720);
    if (fps < 1 || fps > 120) throw std::runtime_error("invalid fps (1..120)");
    validateVideoOutputSize(W, H);
    const double kTop_start = kTopStartForFrame(W, H);
    const double depth      = resolveDepthOctaves(j, kTop_start, 20.0);
    const double secondsPerOctave = resolveSecondsPerOctave(j, depth);
    double durSec = 0.0;
    const int frameCount = frameCountFromSpeed(depth, secondsPerOctave, fps, durSec);
    const StripPlan stripPlan = resolveStripPlan(j, W, H, depth);
    const int s = stripPlan.actualWidthS;

    if (s < 128 || s > 65536)               throw std::runtime_error("invalid widthS (128..65536)");
    if (depth < 0.05 || depth > 120.0)      throw std::runtime_error("invalid depthOctaves (0.05..120)");
    if (iters < 1 || iters > 10000000)      throw std::runtime_error("invalid iterations");

    compute::Variant v;
    if (!compute::variant_from_name(variantStr.c_str(), v)) v = compute::Variant::Mandelbrot;
    double bailout = j.contains("bailout") && !j["bailout"].is_null()
        ? j.value("bailout", 2.0)
        : compute::variant_default_bailout(v);
    const double bailoutSq = bailoutSqFromJson(j, bailout, compute::variant_default_bailout_sq(v));
    if (j.contains("bailoutSq") && !j["bailoutSq"].is_null() &&
        !(j.contains("bailout") && !j["bailout"].is_null())) {
        bailout = std::sqrt(bailoutSq);
    }
    if (!(bailout > 0.0) || !std::isfinite(bailout)) throw std::runtime_error("invalid bailout");
    if (!(bailoutSq > 0.0) || !std::isfinite(bailoutSq)) throw std::runtime_error("invalid bailoutSq");
    compute::Colormap cm;
    if (!compute::colormap_from_name(colormapStr.c_str(), cm)) cm = compute::Colormap::ClassicCos;

    auto run = runner.createRun("video-export", body);

    auto execute = [=, &runner]() mutable -> Json {
    runner.setStatus(run.id, "running");
    try {
        const auto t0 = std::chrono::steady_clock::now();

        // ── 1. Render final cartesian frame ────────────────────────────────────
        setVideoProgress(runner, run.id, "final_frame", 0, 1, depth, depth);
        const double kTop_end   = kTop_start - depth * LN_TWO;

        compute::MapParams mp;
        mp.center_re  = cr;
        mp.center_im  = ci;
        mp.scale      = 2.0 * std::exp(kTop_end);
        mp.width      = W;
        mp.height     = H;
        mp.iterations = iters;
        mp.bailout    = bailout;
        mp.bailout_sq = bailoutSq;
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
        const compute::MapStats finalStats = compute::render_map(mp, finalImg);

        const std::filesystem::path finalPath = std::filesystem::path(run.outputDir) / "final_frame.png";
        compute::write_png(finalPath.string(), finalImg);
        runner.addArtifact(run.id, Artifact{"final-frame", finalPath.string(), "image"});
        setVideoProgress(runner, run.id, "final_frame", 1, 1, depth, depth);

        // ── 2. Render ln-map strip ─────────────────────────────────────────────
        const int t = stripPlan.heightT;
        cv::Mat strip(t, s, CV_8UC3);
        compute::LnMapParams lp;
        lp.julia = julia;
        lp.center_re = cr;
        lp.center_im = ci;
        lp.julia_re = jre;
        lp.julia_im = jim;
        lp.width_s = s;
        lp.height_t = t;
        lp.iterations = iters;
        lp.bailout = bailout;
        lp.bailout_sq = bailoutSq;
        lp.variant = v;
        lp.colormap = cm;
        lp.engine = j.value("lnMapEngine", std::string("auto"));
        setVideoProgress(runner, run.id, "ln_map", 0, t, 0.0, depth);
        const compute::LnMapStats lnStats = compute::render_ln_map(
            lp, strip,
            [&](int rowsDone) {
                const double octave = depth * static_cast<double>(rowsDone) / std::max(1, t);
                setVideoProgress(runner, run.id, "ln_map", rowsDone, t, octave, depth);
            });

        const std::filesystem::path stripPath = std::filesystem::path(run.outputDir) / "ln_map.png";
        compute::write_png(stripPath.string(), strip);
        runner.addArtifact(run.id, Artifact{"ln-map", stripPath.string(), "image"});

        // Sidecar so old zoomVideoRoute can also consume this ln-map.
        {
            Json sc = {
                {"centerRe", cr}, {"centerIm", ci},
                {"julia", julia}, {"juliaRe", jre}, {"juliaIm", jim},
                {"widthS", s}, {"actualWidthS", s}, {"fullWidthS", stripPlan.fullWidthS},
                {"heightT", t}, {"depthOctaves", depth},
                {"qualityPreset", stripPlan.qualityPreset},
                {"qualityScale", stripPlan.qualityScale},
                {"estimatedPeakMemory", stripPlan.estimatedPeakMemory},
                {"lnRadiusTop", LN_FOUR}, {"variant", variantStr},
                {"colorMap", colormapStr}, {"iterations", iters},
                {"bailout", bailout},
                {"bailoutSq", bailoutSq},
                {"engine", lnStats.engine_used},
                {"scalar", lnStats.scalar_used},
            };
            const std::filesystem::path scPath = std::filesystem::path(run.outputDir) / "ln_map.json";
            std::ofstream os(scPath);
            os << sc.dump(2);
        }

        // ── 3. Render first/last preview frames ───────────────────────────────
        const cv::Mat startPreview = renderZoomPreviewFrame(strip, finalImg, W, H, kTop_start, kTop_end);
        const cv::Mat endPreview   = renderZoomPreviewFrame(strip, finalImg, W, H, kTop_end,   kTop_end);
        const std::filesystem::path startPreviewPath = std::filesystem::path(run.outputDir) / "start_frame.png";
        const std::filesystem::path endPreviewPath   = std::filesystem::path(run.outputDir) / "end_frame.png";
        compute::write_png(startPreviewPath.string(), startPreview);
        compute::write_png(endPreviewPath.string(), endPreview);
        runner.addArtifact(run.id, Artifact{"start-frame", startPreviewPath.string(), "image"});
        runner.addArtifact(run.id, Artifact{"end-frame",   endPreviewPath.string(),   "image"});

        // ── 4. Generate video ──────────────────────────────────────────────────
        setVideoProgress(runner, run.id, "video_warp_encode", 0, frameCount, depth, depth);
        std::string ffmpegStderr;
        std::string encoderUsed;
        std::string warpMethod;
        const std::string videoPath = generateZoomVideo(
            strip, finalImg, W, H, fps, frameCount,
            kTop_start, kTop_end, depth,
            std::filesystem::path(run.outputDir), "zoom",
            [&](int frameDone) {
                setVideoProgress(runner, run.id, "video_warp_encode", frameDone, frameCount, depth, depth);
            },
            &ffmpegStderr,
            &encoderUsed,
            j.value("cudaWarp", true),
            &warpMethod
        );
        const std::string videoFile = std::filesystem::path(videoPath).filename().string();
        runner.addArtifact(run.id, Artifact{"zoom-video", videoPath, "video"});

        Json renderLog = {
            {"finalFrameEngine", finalStats.engine_used},
            {"finalFrameScalar", finalStats.scalar_used},
            {"lnMapEngine", lnStats.engine_used},
            {"lnMapScalar", lnStats.scalar_used},
            {"warpMethod", warpMethod},
            {"encoder", encoderUsed},
        };
        const std::filesystem::path reportPath = std::filesystem::path(run.outputDir) / "video_export.json";
        {
            std::ofstream os(reportPath);
            os << renderLog.dump(2);
        }
        runner.addArtifact(run.id, Artifact{"video-export", reportPath.string(), "report"});

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

        runner.setStatus(run.id, "completed");

        const std::string lnArtId    = run.id + ":ln_map.png";
        const std::string finalArtId = run.id + ":final_frame.png";
        const std::string startArtId = run.id + ":start_frame.png";
        const std::string endArtId   = run.id + ":end_frame.png";
        const std::string videoArtId = run.id + ":" + videoFile;
        const std::string reportArtId = run.id + ":video_export.json";

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
            {"startFrameArtifactId", startArtId},
            {"startFrameUrl",      "/api/artifacts/content?artifactId="  + startArtId},
            {"startFrameDownloadUrl", "/api/artifacts/download?artifactId=" + startArtId},
            {"endFrameArtifactId", endArtId},
            {"endFrameUrl",        "/api/artifacts/content?artifactId="  + endArtId},
            {"endFrameDownloadUrl", "/api/artifacts/download?artifactId=" + endArtId},
            {"reportArtifactId",   reportArtId},
            {"reportDownloadUrl",  "/api/artifacts/download?artifactId=" + reportArtId},
            {"frameCount",         frameCount},
            {"fps",                fps},
            {"durationSec",        durSec},
            {"secondsPerOctave",   secondsPerOctave},
            {"depthOctaves",       depth},
            {"targetScale",        2.0 * std::exp(kTop_end)},
            {"fullWidthS",         stripPlan.fullWidthS},
            {"actualWidthS",       s},
            {"heightT",            t},
            {"qualityPreset",      stripPlan.qualityPreset},
            {"qualityScale",       stripPlan.qualityScale},
            {"estimatedPeakMemory", stripPlan.estimatedPeakMemory},
            {"width",              W},
            {"height",             H},
            {"generatedMs",        elapsed},
            {"finalFrameEngine",   finalStats.engine_used},
            {"finalFrameScalar",   finalStats.scalar_used},
            {"lnMapEngine",        lnStats.engine_used},
            {"lnMapScalar",        lnStats.scalar_used},
            {"warpMethod",         warpMethod},
            {"encoder",            encoderUsed},
            {"ffmpegStderr",       ffmpegStderr},
        };
        return resp;

    } catch (const std::exception& e) {
        setVideoProgress(runner, run.id, "failed", 0, 0, 0.0, depth, "video_export", e.what());
        runner.setStatus(run.id, "failed");
        throw;
    }
    };

    const bool background = j.value("background", true);
    if (background) {
        setVideoProgress(runner, run.id, "queued", 0, frameCount, 0.0, depth);
        std::thread([execute]() mutable {
            try {
                (void)execute();
            } catch (...) {}
        }).detach();

        Json resp = {
            {"runId", run.id},
            {"status", "queued"},
            {"frameCount", frameCount},
            {"fps", fps},
            {"durationSec", durSec},
            {"secondsPerOctave", secondsPerOctave},
            {"depthOctaves", depth},
            {"fullWidthS", stripPlan.fullWidthS},
            {"actualWidthS", s},
            {"heightT", stripPlan.heightT},
            {"qualityPreset", stripPlan.qualityPreset},
            {"qualityScale", stripPlan.qualityScale},
            {"estimatedPeakMemory", stripPlan.estimatedPeakMemory},
            {"width", W},
            {"height", H},
            {"lnMapEngine", "openmp"},
            {"lnMapScalar", "fp64"},
            {"warpMethod", "opencv_cpu_remap"},
        };
        return resp.dump();
    }

    return execute().dump();
}

} // namespace fsd
