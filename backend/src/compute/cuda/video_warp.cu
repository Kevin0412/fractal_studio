// compute/cuda/video_warp.cu

#include "video_warp.cuh"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_WARP_CHECK(expr) do {                                           \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess)                                                   \
        throw std::runtime_error(std::string("CUDA video warp: ") + cudaGetErrorString(_e)); \
} while (0)

namespace fsd_cuda {
namespace {

constexpr double TAU = 6.283185307179586;
constexpr double LN_FOUR = 1.3862943611198906;

__device__ inline double clamp01(double v) {
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

__device__ inline void sample_bgr(
    const unsigned char* img,
    int width,
    int height,
    double x,
    double y,
    double& b,
    double& g,
    double& r
) {
    b = g = r = 0.0;
    const int x0 = static_cast<int>(floor(x));
    const int y0 = static_cast<int>(floor(y));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    for (int oy = 0; oy <= 1; ++oy) {
        const int sy = y0 + oy;
        if (sy < 0 || sy >= height) continue;
        const double wy = oy == 0 ? (1.0 - fy) : fy;
        for (int ox = 0; ox <= 1; ++ox) {
            const int sx = x0 + ox;
            if (sx < 0 || sx >= width) continue;
            const double wx = ox == 0 ? (1.0 - fx) : fx;
            const double w = wx * wy;
            const unsigned char* px = img + 3 * (sy * width + sx);
            b += w * static_cast<double>(px[0]);
            g += w * static_cast<double>(px[1]);
            r += w * static_cast<double>(px[2]);
        }
    }
}

__global__ void warp_kernel(
    const unsigned char* strip,
    const unsigned char* finalImg,
    unsigned char* out,
    int W,
    int H,
    int stripW,
    int stripH,
    double kTop,
    double kTopEnd
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H;
    if (idx >= total) return;
    const int x = idx % W;
    const int y = idx / W;
    const double aspect = static_cast<double>(W) / static_cast<double>(H);
    const int s = stripW - 1;
    const double ux = (2.0 * (static_cast<double>(x) + 0.5) / static_cast<double>(W) - 1.0) * aspect;
    const double vy = -(2.0 * (static_cast<double>(y) + 0.5) / static_cast<double>(H) - 1.0);
    const double r2 = ux * ux + vy * vy;
    double th = atan2(vy, ux);
    if (th < 0.0) th += TAU;

    bool useStrip = false;
    double stripX = 0.0;
    double stripY = 0.0;
    if (r2 > 1e-30) {
        const double lnR = 0.5 * log(r2);
        stripY = (LN_FOUR - kTop - lnR) * static_cast<double>(s) / TAU;
        stripX = th / TAU * static_cast<double>(s);
        useStrip = stripY >= 0.0 && stripY < static_cast<double>(stripH) - 1.0;
    }

    double b = 0.0, g = 0.0, r = 0.0;
    if (useStrip) {
        sample_bgr(strip, stripW, stripH, stripX, stripY, b, g, r);
    } else {
        const double S = exp(kTop - kTopEnd);
        const double fu = ux * S;
        const double fv = vy * S;
        const double finalX = (fu / aspect * 0.5 + 0.5) * static_cast<double>(W);
        const double finalY = (-fv * 0.5 + 0.5) * static_cast<double>(H);
        sample_bgr(finalImg, W, H, finalX, finalY, b, g, r);
    }

    unsigned char* dp = out + 3 * idx;
    dp[0] = static_cast<unsigned char>(clamp01(b / 255.0) * 255.0 + 0.5);
    dp[1] = static_cast<unsigned char>(clamp01(g / 255.0) * 255.0 + 0.5);
    dp[2] = static_cast<unsigned char>(clamp01(r / 255.0) * 255.0 + 0.5);
}

} // namespace

bool cuda_video_warp_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void cuda_video_warp_init(const cv::Mat& stripWrap, const cv::Mat& finalImg, CudaVideoWarpContext& ctx) {
    if (!cuda_video_warp_available()) throw std::runtime_error("CUDA video warp not available");
    if (stripWrap.empty() || finalImg.empty() || stripWrap.type() != CV_8UC3 || finalImg.type() != CV_8UC3) {
        throw std::runtime_error("CUDA video warp expects CV_8UC3 inputs");
    }
    cuda_video_warp_release(ctx);
    ctx.width = finalImg.cols;
    ctx.height = finalImg.rows;
    ctx.strip_width = stripWrap.cols;
    ctx.strip_height = stripWrap.rows;
    const size_t stripBytes = static_cast<size_t>(ctx.strip_width) * ctx.strip_height * 3u;
    const size_t finalBytes = static_cast<size_t>(ctx.width) * ctx.height * 3u;
    CUDA_WARP_CHECK(cudaMalloc(&ctx.d_strip, stripBytes));
    CUDA_WARP_CHECK(cudaMalloc(&ctx.d_final, finalBytes));
    CUDA_WARP_CHECK(cudaMalloc(&ctx.d_out, finalBytes));
    CUDA_WARP_CHECK(cudaMemcpy(ctx.d_strip, stripWrap.data, stripBytes, cudaMemcpyHostToDevice));
    CUDA_WARP_CHECK(cudaMemcpy(ctx.d_final, finalImg.data, finalBytes, cudaMemcpyHostToDevice));
}

void cuda_video_warp_frame(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame) {
    if (!ctx.d_strip || !ctx.d_final || !ctx.d_out) throw std::runtime_error("CUDA video warp context not initialized");
    if (frame.empty() || frame.rows != ctx.height || frame.cols != ctx.width || frame.type() != CV_8UC3) {
        frame.create(ctx.height, ctx.width, CV_8UC3);
    }
    const int total = ctx.width * ctx.height;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    warp_kernel<<<grid, block>>>(
        static_cast<const unsigned char*>(ctx.d_strip),
        static_cast<const unsigned char*>(ctx.d_final),
        static_cast<unsigned char*>(ctx.d_out),
        ctx.width, ctx.height, ctx.strip_width, ctx.strip_height,
        kTop, kTopEnd);
    CUDA_WARP_CHECK(cudaGetLastError());
    const size_t bytes = static_cast<size_t>(ctx.width) * ctx.height * 3u;
    CUDA_WARP_CHECK(cudaMemcpy(frame.data, ctx.d_out, bytes, cudaMemcpyDeviceToHost));
}

void cuda_video_warp_release(CudaVideoWarpContext& ctx) noexcept {
    if (ctx.d_strip) cudaFree(ctx.d_strip);
    if (ctx.d_final) cudaFree(ctx.d_final);
    if (ctx.d_out) cudaFree(ctx.d_out);
    ctx = CudaVideoWarpContext{};
}

} // namespace fsd_cuda
