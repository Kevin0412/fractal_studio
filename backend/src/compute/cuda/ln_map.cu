// compute/cuda/ln_map.cu

#include "ln_map.cuh"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

#define CUDA_LN_CHECK(expr) do {                                             \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess)                                                   \
        throw std::runtime_error(std::string("CUDA ln-map: ") + cudaGetErrorString(_e)); \
} while (0)

namespace fsd_cuda {
namespace {

constexpr double TAU = 6.283185307179586;
constexpr double LN_FOUR = 1.3862943611198906;

std::mutex g_ln_mu;

__device__ inline int d_clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__device__ inline double d_cos_color(double n, double freq) {
    constexpr double PI = 3.141592653589793;
    return 128.0 - 128.0 * cos(freq * n * PI);
}

__device__ inline void d_hsv_to_rgb(double h, double s, double v, int& r, int& g, int& b) {
    const double c = v * s;
    const double hh = h / 60.0;
    const double x = c * (1.0 - fabs(fmod(hh, 2.0) - 1.0));
    double rr = 0.0, gg = 0.0, bb = 0.0;
    if      (hh < 1.0) { rr = c; gg = x; }
    else if (hh < 2.0) { rr = x; gg = c; }
    else if (hh < 3.0) { gg = c; bb = x; }
    else if (hh < 4.0) { gg = x; bb = c; }
    else if (hh < 5.0) { rr = x; bb = c; }
    else               { rr = c; bb = x; }
    const double m = v - c;
    r = d_clamp255(static_cast<int>((rr + m) * 255.0));
    g = d_clamp255(static_cast<int>((gg + m) * 255.0));
    b = d_clamp255(static_cast<int>((bb + m) * 255.0));
}

__device__ inline void d_colorize_escape_bgr(int iter, int max_iter, int cmap, unsigned char* px) {
    if (iter >= max_iter) {
        px[0] = px[1] = px[2] = 255;
        return;
    }
    const double n = (static_cast<double>(iter) + 1.0) / (static_cast<double>(max_iter) + 2.0);
    switch (cmap) {
        case 1:
            px[2] = static_cast<unsigned char>(d_clamp255(iter % 256));
            px[1] = static_cast<unsigned char>(d_clamp255(iter / 256));
            px[0] = static_cast<unsigned char>(d_clamp255((iter % 17) * 17));
            return;
        case 2: {
            const double h = fmod(static_cast<double>(iter), 1440.0) / 4.0;
            int r = 0, g = 0, b = 0;
            d_hsv_to_rgb(h, 1.0, 1.0, r, g, b);
            px[2] = static_cast<unsigned char>(r);
            px[1] = static_cast<unsigned char>(g);
            px[0] = static_cast<unsigned char>(b);
            return;
        }
        case 3: {
            const int m = iter % 765;
            const int band = m / 255;
            const int d = m % 255;
            int r = 255, g = 255, b = 255;
            if      (band == 0) { r = 255 - d; g = d;       b = 255;     }
            else if (band == 1) { r = d;       g = 255;     b = 255 - d; }
            else                { r = 255;     g = 255 - d; b = d;       }
            px[2] = static_cast<unsigned char>(d_clamp255(r));
            px[1] = static_cast<unsigned char>(d_clamp255(g));
            px[0] = static_cast<unsigned char>(d_clamp255(b));
            return;
        }
        case 4: {
            const int v = d_clamp255(static_cast<int>(n * 255.0));
            px[0] = px[1] = px[2] = static_cast<unsigned char>(v);
            return;
        }
        default:
            px[2] = static_cast<unsigned char>(d_clamp255(static_cast<int>(d_cos_color(n, 53.0))));
            px[1] = static_cast<unsigned char>(d_clamp255(static_cast<int>(d_cos_color(n, 27.0))));
            px[0] = static_cast<unsigned char>(d_clamp255(static_cast<int>(d_cos_color(n, 139.0))));
            return;
    }
}

__device__ inline void d_step(int variant, double zr, double zi, double cr, double ci, double& nr, double& ni) {
    const double zr2 = zr * zr;
    const double zi2 = zi * zi;
    const double sq_re = zr2 - zi2;
    const double sq_im = 2.0 * zr * zi;
    switch (variant) {
        case 1: nr = sq_re + cr; ni = ci - sq_im; break;
        case 2: nr = sq_re + cr; ni = 2.0 * fabs(zr) * fabs(zi) + ci; break;
        case 3: nr = sq_re + cr; ni = 2.0 * zr * fabs(zi) + ci; break;
        case 4: nr = sq_re + cr; ni = -2.0 * fabs(zr) * zi + ci; break;
        case 5: nr = fabs(sq_re) + cr; ni = sq_im + ci; break;
        case 6: nr = fabs(sq_re) + cr; ni = ci - sq_im; break;
        case 7: nr = fabs(sq_re) + cr; ni = fabs(sq_im) + ci; break;
        case 8: nr = fabs(sq_re) + cr; ni = 2.0 * zr * fabs(zi) + ci; break;
        case 9: nr = fabs(sq_re) + cr; ni = ci - 2.0 * fabs(zr) * zi; break;
        default: nr = sq_re + cr; ni = sq_im + ci; break;
    }
}

__global__ void ln_map_kernel(CudaLnMapParams p, unsigned char* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = p.width_s * p.height_t;
    if (idx >= total) return;
    const int x = idx % p.width_s;
    const int row = idx / p.width_s;

    const double th = TAU * static_cast<double>(x) / static_cast<double>(p.width_s);
    const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(p.width_s);
    const double r_mag = exp(k);
    const double pre = p.center_re + r_mag * cos(th);
    const double pim = p.center_im + r_mag * sin(th);

    double zr = p.julia ? pre : 0.0;
    double zi = p.julia ? pim : 0.0;
    const double cr = p.julia ? p.julia_re : pre;
    const double ci = p.julia ? p.julia_im : pim;
    int iter = 0;
    for (; iter < p.iterations; ++iter) {
        double nr = 0.0, ni = 0.0;
        d_step(p.variant_id, zr, zi, cr, ci, nr, ni);
        zr = nr;
        zi = ni;
        const double norm2 = zr * zr + zi * zi;
        if (!isfinite(norm2) || norm2 > p.bailout_sq) break;
    }

    d_colorize_escape_bgr(iter, p.iterations, p.colormap_id, out + 3 * idx);
}

} // namespace

bool cuda_ln_map_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaLnMapStats cuda_render_ln_map(const CudaLnMapParams& p, cv::Mat& out) {
    if (!cuda_ln_map_available()) throw std::runtime_error("CUDA ln-map not available");
    std::lock_guard<std::mutex> lock(g_ln_mu);

    if (out.empty() || out.rows != p.height_t || out.cols != p.width_s || out.type() != CV_8UC3) {
        out.create(p.height_t, p.width_s, CV_8UC3);
    }
    const size_t bytes = static_cast<size_t>(p.width_s) * p.height_t * 3u;
    unsigned char* d_out = nullptr;
    CUDA_LN_CHECK(cudaMalloc(&d_out, bytes));

    cudaEvent_t start, stop;
    CUDA_LN_CHECK(cudaEventCreate(&start));
    CUDA_LN_CHECK(cudaEventCreate(&stop));
    CUDA_LN_CHECK(cudaEventRecord(start));

    const int block = 256;
    const int total = p.width_s * p.height_t;
    const int grid = (total + block - 1) / block;
    ln_map_kernel<<<grid, block>>>(p, d_out);
    CUDA_LN_CHECK(cudaGetLastError());
    CUDA_LN_CHECK(cudaEventRecord(stop));
    CUDA_LN_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_LN_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_LN_CHECK(cudaMemcpy(out.data, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_LN_CHECK(cudaEventDestroy(start));
    CUDA_LN_CHECK(cudaEventDestroy(stop));
    cudaFree(d_out);

    CudaLnMapStats stats;
    stats.elapsed_ms = static_cast<double>(ms);
    return stats;
}

} // namespace fsd_cuda
