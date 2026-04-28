// compute/cuda/ln_map.cu

#include "ln_map.cuh"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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

template <int VariantId>
__device__ inline void d_step_cached(
    double zr,
    double zi,
    double zr2,
    double zi2,
    double cr,
    double ci,
    double& nr,
    double& ni
) {
    const double sq_re = zr2 - zi2;
    const double sq_im = 2.0 * zr * zi;
    if constexpr (VariantId == 1) {
        nr = sq_re + cr;
        ni = ci - sq_im;
    } else if constexpr (VariantId == 2) {
        nr = sq_re + cr;
        ni = 2.0 * fabs(zr) * fabs(zi) + ci;
    } else if constexpr (VariantId == 3) {
        nr = sq_re + cr;
        ni = 2.0 * zr * fabs(zi) + ci;
    } else if constexpr (VariantId == 4) {
        nr = sq_re + cr;
        ni = -2.0 * fabs(zr) * zi + ci;
    } else if constexpr (VariantId == 5) {
        nr = fabs(sq_re) + cr;
        ni = sq_im + ci;
    } else if constexpr (VariantId == 6) {
        nr = fabs(sq_re) + cr;
        ni = ci - sq_im;
    } else if constexpr (VariantId == 7) {
        nr = fabs(sq_re) + cr;
        ni = fabs(sq_im) + ci;
    } else if constexpr (VariantId == 8) {
        nr = fabs(sq_re) + cr;
        ni = 2.0 * zr * fabs(zi) + ci;
    } else if constexpr (VariantId == 9) {
        nr = fabs(sq_re) + cr;
        ni = ci - 2.0 * fabs(zr) * zi;
    } else {
        nr = sq_re + cr;
        ni = sq_im + ci;
    }
}

template <int VariantId>
__global__ void ln_map_kernel_templated(CudaLnMapParams p, int row_start, int row_count, unsigned char* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = p.width_s * row_count;
    if (idx >= total) return;
    const int x = idx % p.width_s;
    const int local_row = idx / p.width_s;
    const int row = row_start + local_row;

    const double th = TAU * static_cast<double>(x) / static_cast<double>(p.width_s);
    const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(p.width_s);
    const double r_mag = exp(k);
    const double pre = p.center_re + r_mag * cos(th);
    const double pim = p.center_im + r_mag * sin(th);

    double zr = p.julia ? pre : 0.0;
    double zi = p.julia ? pim : 0.0;
    const double cr = p.julia ? p.julia_re : pre;
    const double ci = p.julia ? p.julia_im : pim;
    double zr2 = zr * zr;
    double zi2 = zi * zi;

    int iter = 0;
    for (; iter < p.iterations; ++iter) {
        double nr = 0.0, ni = 0.0;
        d_step_cached<VariantId>(zr, zi, zr2, zi2, cr, ci, nr, ni);
        const bool finite = isfinite(nr) && isfinite(ni);
        const double nr2 = finite ? nr * nr : INFINITY;
        const double ni2 = finite ? ni * ni : INFINITY;
        const double norm2 = nr2 + ni2;
        if (!finite || norm2 > p.bailout_sq) break;
        zr = nr;
        zi = ni;
        zr2 = nr2;
        zi2 = ni2;
    }

    d_colorize_escape_bgr(iter, p.iterations, p.colormap_id, out + 3 * idx);
}

template <int VariantId>
void launch_ln_map_variant(const CudaLnMapParams& p, int row_start, int row_count, unsigned char* d_out) {
    const int block = 256;
    const int total = p.width_s * row_count;
    const int grid = (total + block - 1) / block;
    ln_map_kernel_templated<VariantId><<<grid, block>>>(p, row_start, row_count, d_out);
}

void launch_ln_map(const CudaLnMapParams& p, int row_start, int row_count, unsigned char* d_out) {
    switch (p.variant_id) {
        case 0: launch_ln_map_variant<0>(p, row_start, row_count, d_out); break;
        case 1: launch_ln_map_variant<1>(p, row_start, row_count, d_out); break;
        case 2: launch_ln_map_variant<2>(p, row_start, row_count, d_out); break;
        case 3: launch_ln_map_variant<3>(p, row_start, row_count, d_out); break;
        case 4: launch_ln_map_variant<4>(p, row_start, row_count, d_out); break;
        case 5: launch_ln_map_variant<5>(p, row_start, row_count, d_out); break;
        case 6: launch_ln_map_variant<6>(p, row_start, row_count, d_out); break;
        case 7: launch_ln_map_variant<7>(p, row_start, row_count, d_out); break;
        case 8: launch_ln_map_variant<8>(p, row_start, row_count, d_out); break;
        case 9: launch_ln_map_variant<9>(p, row_start, row_count, d_out); break;
        default: throw std::runtime_error("CUDA ln-map unsupported variant");
    }
}

void ensure_out(const CudaLnMapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height_t || out.cols != p.width_s || out.type() != CV_8UC3) {
        out.create(p.height_t, p.width_s, CV_8UC3);
    }
}

} // namespace

bool cuda_ln_map_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaLnMapStats cuda_render_ln_map_rows(const CudaLnMapParams& p, cv::Mat& out, int row_start, int row_count) {
    if (!cuda_ln_map_available()) throw std::runtime_error("CUDA ln-map not available");
    if (p.variant_id < 0 || p.variant_id > 9) throw std::runtime_error("CUDA ln-map unsupported variant");
    if (row_start < 0 || row_count <= 0 || row_start + row_count > p.height_t) {
        throw std::runtime_error("invalid CUDA ln-map row range");
    }
    std::lock_guard<std::mutex> lock(g_ln_mu);
    ensure_out(p, out);

    const size_t row_bytes = static_cast<size_t>(p.width_s) * 3u;
    const size_t bytes = row_bytes * static_cast<size_t>(row_count);
    unsigned char* d_out = nullptr;
    CUDA_LN_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));

    cudaEvent_t start, stop;
    CUDA_LN_CHECK(cudaEventCreate(&start));
    CUDA_LN_CHECK(cudaEventCreate(&stop));
    CUDA_LN_CHECK(cudaEventRecord(start));
    launch_ln_map(p, row_start, row_count, d_out);
    CUDA_LN_CHECK(cudaGetLastError());
    CUDA_LN_CHECK(cudaEventRecord(stop));
    CUDA_LN_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_LN_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (out.isContinuous() && out.step == row_bytes) {
        CUDA_LN_CHECK(cudaMemcpy(out.ptr<unsigned char>(row_start), d_out, bytes, cudaMemcpyDeviceToHost));
    } else {
        std::vector<unsigned char> tmp(bytes);
        CUDA_LN_CHECK(cudaMemcpy(tmp.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        for (int r = 0; r < row_count; ++r) {
            std::memcpy(out.ptr<unsigned char>(row_start + r), tmp.data() + static_cast<size_t>(r) * row_bytes, row_bytes);
        }
    }
    CUDA_LN_CHECK(cudaEventDestroy(start));
    CUDA_LN_CHECK(cudaEventDestroy(stop));
    cudaFree(d_out);

    CudaLnMapStats stats;
    stats.elapsed_ms = static_cast<double>(ms);
    return stats;
}

CudaLnMapStats cuda_render_ln_map(const CudaLnMapParams& p, cv::Mat& out) {
    return cuda_render_ln_map_rows(p, out, 0, p.height_t);
}

} // namespace fsd_cuda
