// compute/cuda/video_warp.cuh

#pragma once

#include <opencv2/core.hpp>

#include <cstdint>

namespace fsd_cuda {

struct CudaVideoWarpContext {
    int width = 0;
    int height = 0;
    int strip_width = 0; // includes one wrap column
    int strip_height = 0;
    void* d_strip = nullptr;
    void* d_final = nullptr;
    void* d_geom = nullptr;
    void* d_out = nullptr;
    void* strip_array = nullptr;
    void* final_array = nullptr;
    uint64_t strip_tex = 0;
    uint64_t final_tex = 0;
    void* kernel_start_event = nullptr;
    void* kernel_stop_event = nullptr;
};

struct CudaVideoWarpTiming {
    double kernel_ms = 0.0;
    double copy_ms = 0.0;
};

bool cuda_video_warp_available() noexcept;
void cuda_video_warp_init(const cv::Mat& stripWrap, const cv::Mat& finalImg, CudaVideoWarpContext& ctx);
void cuda_video_warp_frame_timed(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame, CudaVideoWarpTiming* timing);
void cuda_video_warp_frame(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame);
void cuda_video_warp_release(CudaVideoWarpContext& ctx) noexcept;

} // namespace fsd_cuda
