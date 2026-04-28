// compute/cuda/transition_volume.cuh

#pragma once

#include <vector>

namespace fsd_cuda {

struct CudaTransitionVolumeParams {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float center_z = 0.0f;
    float extent = 2.0f;
    int resolution = 96;
    int iterations = 256;
    float bailout = 2.0f;
    float bailout_sq = 4.0f;
    int from_variant = 0;
    int to_variant = 2;
};

bool cuda_transition_available() noexcept;
void cuda_build_transition_volume(const CudaTransitionVolumeParams& p, std::vector<float>& out);
void cuda_build_transition_volume_slabs(const CudaTransitionVolumeParams& p, int z_start, int z_count, std::vector<float>& out);

} // namespace fsd_cuda
