// compute/cuda/video_warp.cu

#include "video_warp.cuh"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_WARP_CHECK(expr) do {                                           \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess)                                                   \
        throw std::runtime_error(std::string("CUDA video warp: ") + cudaGetErrorString(_e)); \
} while (0)

namespace fsd_cuda {
namespace {

constexpr float TAU_F = 6.2831853071795864769f;
constexpr float LN_FOUR_F = 1.3862943611198906f;

struct WarpGeom {
    float strip_x;
    float strip_row_base;
    float final_x_base;
    float final_y_base;
    unsigned char valid_r;
    unsigned char pad[3];
};

static_assert(sizeof(WarpGeom) == 20, "WarpGeom layout changed");

__device__ inline unsigned char float_to_u8(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return static_cast<unsigned char>(v * 255.0f + 0.5f);
}

__global__ void precompute_geom_kernel(WarpGeom* geom, int W, int H, int stripW) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H;
    if (idx >= total) return;

    const int x = idx % W;
    const int y = idx / W;
    const float aspect = static_cast<float>(W) / static_cast<float>(H);
    const int s = stripW - 1;
    const float stripScale = static_cast<float>(s) / TAU_F;

    const float ux = (2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(W) - 1.0f) * aspect;
    const float vy = -(2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(H) - 1.0f);
    const float r2 = ux * ux + vy * vy;

    float theta = atan2f(vy, ux);
    if (theta < 0.0f) theta += TAU_F;

    WarpGeom g{};
    g.strip_x = theta * stripScale;
    g.final_x_base = (ux / aspect) * 0.5f * static_cast<float>(W);
    g.final_y_base = (-vy) * 0.5f * static_cast<float>(H);
    if (r2 > 1.0e-30f) {
        const float lnR = 0.5f * logf(r2);
        g.strip_row_base = (LN_FOUR_F - lnR) * stripScale;
        g.valid_r = 1;
    } else {
        g.strip_row_base = -1.0f;
        g.valid_r = 0;
    }
    geom[idx] = g;
}

__global__ void warp_texture_kernel(
    cudaTextureObject_t stripTex,
    cudaTextureObject_t finalTex,
    const WarpGeom* geom,
    unsigned char* out,
    int W,
    int H,
    int stripW,
    int stripH,
    float kTopStripScale,
    float finalScale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H;
    if (idx >= total) return;

    const WarpGeom g = geom[idx];
    const float stripY = g.strip_row_base - kTopStripScale;
    float4 color;
    if (g.valid_r && stripY >= 0.0f && stripY < static_cast<float>(stripH - 1)) {
        const float stripU = (g.strip_x + 0.5f) / static_cast<float>(stripW);
        const float stripV = (stripY + 0.5f) / static_cast<float>(stripH);
        color = tex2D<float4>(stripTex, stripU, stripV);
    } else {
        const float finalX = g.final_x_base * finalScale + 0.5f * static_cast<float>(W);
        const float finalY = g.final_y_base * finalScale + 0.5f * static_cast<float>(H);
        color = tex2D<float4>(finalTex, finalX + 1.5f, finalY + 1.5f);
    }

    unsigned char* dp = out + 3 * idx;
    dp[0] = float_to_u8(color.x);
    dp[1] = float_to_u8(color.y);
    dp[2] = float_to_u8(color.z);
}

void upload_bgra_array(const cv::Mat& bgra, void*& arrayOut, uint64_t& texOut, bool wrapX, bool normalizedCoords) {
    cudaArray_t array = nullptr;
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    CUDA_WARP_CHECK(cudaMallocArray(&array, &channelDesc, bgra.cols, bgra.rows));
    try {
        CUDA_WARP_CHECK(cudaMemcpy2DToArray(
            array, 0, 0,
            bgra.data, bgra.step,
            static_cast<size_t>(bgra.cols) * sizeof(uchar4),
            bgra.rows,
            cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = wrapX ? cudaAddressModeWrap : cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = normalizedCoords ? 1 : 0;

        cudaTextureObject_t tex = 0;
        CUDA_WARP_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
        arrayOut = array;
        texOut = static_cast<uint64_t>(tex);
    } catch (...) {
        cudaFreeArray(array);
        throw;
    }
}

void init_events(CudaVideoWarpContext& ctx) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_WARP_CHECK(cudaEventCreate(&start));
    try {
        CUDA_WARP_CHECK(cudaEventCreate(&stop));
    } catch (...) {
        cudaEventDestroy(start);
        throw;
    }
    ctx.kernel_start_event = start;
    ctx.kernel_stop_event = stop;
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

    cv::Mat stripBgra;
    cv::cvtColor(stripWrap, stripBgra, cv::COLOR_BGR2BGRA);

    cv::Mat finalPadded;
    cv::copyMakeBorder(finalImg, finalPadded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat finalBgra;
    cv::cvtColor(finalPadded, finalBgra, cv::COLOR_BGR2BGRA);

    const size_t frameBytes = static_cast<size_t>(ctx.width) * ctx.height * 3u;
    const size_t geomBytes = static_cast<size_t>(ctx.width) * ctx.height * sizeof(WarpGeom);

    try {
        upload_bgra_array(stripBgra, ctx.strip_array, ctx.strip_tex, true, true);
        upload_bgra_array(finalBgra, ctx.final_array, ctx.final_tex, false, false);
        CUDA_WARP_CHECK(cudaMalloc(&ctx.d_geom, geomBytes));
        CUDA_WARP_CHECK(cudaMalloc(&ctx.d_out, frameBytes));
        init_events(ctx);

        const int total = ctx.width * ctx.height;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        precompute_geom_kernel<<<grid, block>>>(
            static_cast<WarpGeom*>(ctx.d_geom),
            ctx.width, ctx.height, ctx.strip_width);
        CUDA_WARP_CHECK(cudaGetLastError());
        CUDA_WARP_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        cuda_video_warp_release(ctx);
        throw;
    }
}

void cuda_video_warp_frame_timed(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame, CudaVideoWarpTiming* timing) {
    if (!ctx.d_geom || !ctx.d_out || !ctx.strip_tex || !ctx.final_tex) {
        throw std::runtime_error("CUDA video warp context not initialized");
    }
    if (frame.empty() || frame.rows != ctx.height || frame.cols != ctx.width || frame.type() != CV_8UC3 || !frame.isContinuous()) {
        frame.create(ctx.height, ctx.width, CV_8UC3);
    }
    const int total = ctx.width * ctx.height;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    const int s = ctx.strip_width - 1;
    const float kTopStripScale = static_cast<float>(kTop * static_cast<double>(s) / 6.2831853071795864769);
    const float finalScale = static_cast<float>(std::exp(kTop - kTopEnd));

    cudaEvent_t kernelStart = static_cast<cudaEvent_t>(ctx.kernel_start_event);
    cudaEvent_t kernelStop = static_cast<cudaEvent_t>(ctx.kernel_stop_event);
    CUDA_WARP_CHECK(cudaEventRecord(kernelStart, 0));
    warp_texture_kernel<<<grid, block>>>(
        static_cast<cudaTextureObject_t>(ctx.strip_tex),
        static_cast<cudaTextureObject_t>(ctx.final_tex),
        static_cast<const WarpGeom*>(ctx.d_geom),
        static_cast<unsigned char*>(ctx.d_out),
        ctx.width, ctx.height, ctx.strip_width, ctx.strip_height,
        kTopStripScale, finalScale);
    CUDA_WARP_CHECK(cudaGetLastError());
    CUDA_WARP_CHECK(cudaEventRecord(kernelStop, 0));
    CUDA_WARP_CHECK(cudaEventSynchronize(kernelStop));

    float kernelMs = 0.0f;
    CUDA_WARP_CHECK(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop));

    const size_t bytes = static_cast<size_t>(ctx.width) * ctx.height * 3u;
    const auto copyStart = std::chrono::steady_clock::now();
    CUDA_WARP_CHECK(cudaMemcpy(frame.data, ctx.d_out, bytes, cudaMemcpyDeviceToHost));
    const auto copyEnd = std::chrono::steady_clock::now();

    if (timing) {
        timing->kernel_ms = static_cast<double>(kernelMs);
        timing->copy_ms = std::chrono::duration<double, std::milli>(copyEnd - copyStart).count();
    }
}

void cuda_video_warp_frame(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame) {
    cuda_video_warp_frame_timed(ctx, kTop, kTopEnd, frame, nullptr);
}

void cuda_video_warp_release(CudaVideoWarpContext& ctx) noexcept {
    if (ctx.d_strip) cudaFree(ctx.d_strip);
    if (ctx.d_final) cudaFree(ctx.d_final);
    if (ctx.d_geom) cudaFree(ctx.d_geom);
    if (ctx.d_out) cudaFree(ctx.d_out);
    if (ctx.strip_tex) cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(ctx.strip_tex));
    if (ctx.final_tex) cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(ctx.final_tex));
    if (ctx.strip_array) cudaFreeArray(static_cast<cudaArray_t>(ctx.strip_array));
    if (ctx.final_array) cudaFreeArray(static_cast<cudaArray_t>(ctx.final_array));
    if (ctx.kernel_start_event) cudaEventDestroy(static_cast<cudaEvent_t>(ctx.kernel_start_event));
    if (ctx.kernel_stop_event) cudaEventDestroy(static_cast<cudaEvent_t>(ctx.kernel_stop_event));
    ctx = CudaVideoWarpContext{};
}

} // namespace fsd_cuda
