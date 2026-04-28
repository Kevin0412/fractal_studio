// compute/cuda/video_warp.cuh

#pragma once

#include <opencv2/core.hpp>

namespace fsd_cuda {

struct CudaVideoWarpContext {
    int width = 0;
    int height = 0;
    int strip_width = 0; // includes one wrap column
    int strip_height = 0;
    void* d_strip = nullptr;
    void* d_final = nullptr;
    void* d_out = nullptr;
};

bool cuda_video_warp_available() noexcept;
void cuda_video_warp_init(const cv::Mat& stripWrap, const cv::Mat& finalImg, CudaVideoWarpContext& ctx);
void cuda_video_warp_frame(CudaVideoWarpContext& ctx, double kTop, double kTopEnd, cv::Mat& frame);
void cuda_video_warp_release(CudaVideoWarpContext& ctx) noexcept;

} // namespace fsd_cuda
