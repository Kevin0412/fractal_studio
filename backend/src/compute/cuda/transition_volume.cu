// compute/cuda/transition_volume.cu

#include "transition_volume.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)  do {                                                \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess)                                                    \
        throw std::runtime_error(std::string("CUDA transition: ") + cudaGetErrorString(_e)); \
} while(0)

namespace fsd_cuda {

__device__ inline float real_projection(int v, float x2, float axis2) {
    const bool post_abs = v == 5 || v == 6 || v == 7 || v == 8 || v == 9;
    float q = x2 - axis2;
    return post_abs ? fabsf(q) : q;
}

__device__ inline float imag_projection(int v, float x, float axis) {
    const bool abs_x = v == 2 || v == 4 || v == 9;
    const bool abs_axis = v == 2 || v == 3 || v == 7 || v == 8;
    const bool neg = v == 1 || v == 4 || v == 6 || v == 9;
    const float a = abs_x ? fabsf(x) : x;
    const float b = abs_axis ? fabsf(axis) : axis;
    const float q = 2.0f * a * b;
    return neg ? -q : q;
}

__global__ void transition_volume_kernel(CudaTransitionVolumeParams p, float* out) {
    const int N = p.resolution;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * N * N;
    if (idx >= total) return;

    const int xi = idx % N;
    const int yi = (idx / N) % N;
    const int zi = idx / (N * N);

    const float span = p.extent * 2.0f;
    const float x0 = (p.center_x - p.extent) + (static_cast<float>(xi) + 0.5f) / static_cast<float>(N) * span;
    const float y0 = (p.center_y - p.extent) + (static_cast<float>(yi) + 0.5f) / static_cast<float>(N) * span;
    const float z0 = (p.center_z - p.extent) + (static_cast<float>(zi) + 0.5f) / static_cast<float>(N) * span;

    float x = x0, y = y0, z = z0;
    int iter = 0;
    bool escaped = false;
    for (; iter < p.iterations; ++iter) {
        const float x2 = x * x;
        const float nx = real_projection(p.from_variant, x2, y * y)
                       + real_projection(p.to_variant,   x2, z * z)
                       - x2 + x0;
        const float ny = imag_projection(p.from_variant, x, y) + y0;
        const float nz = imag_projection(p.to_variant,   x, z) + z0;
        x = nx; y = ny; z = nz;
        const float n2 = x * x + y * y + z * z;
        if (!isfinite(n2) || n2 > p.bailout_sq) {
            escaped = true;
            break;
        }
    }

    if (escaped) {
        out[idx] = 0.5f + 0.5f * (static_cast<float>(iter) / static_cast<float>(p.iterations));
    } else {
        const float final_mag = sqrtf(x * x + y * y + z * z);
        out[idx] = fminf(0.48f, final_mag / p.bailout * 0.48f);
    }
}

bool cuda_transition_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void cuda_build_transition_volume(const CudaTransitionVolumeParams& p, std::vector<float>& out) {
    if (!cuda_transition_available()) throw std::runtime_error("CUDA transition not available");
    const size_t count = static_cast<size_t>(p.resolution) * p.resolution * p.resolution;
    out.assign(count, 1.0f);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    transition_volume_kernel<<<grid, block>>>(p, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
}

} // namespace fsd_cuda
