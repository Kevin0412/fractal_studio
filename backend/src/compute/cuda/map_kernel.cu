// compute/cuda/map_kernel.cu
//
// CUDA fractal renderer supporting all 10 variants, Julia mode,
// and non-escape metrics (MinAbs, MaxAbs, Envelope).
//
// Design:
//   - One kernel handles all pixels via a flat thread grid (pixel = thread).
//   - Variant and metric are selected at host-call time and compiled into
//     templated fp64/fx64 kernels.
//   - Output: raw BGR byte array copied into a cv::Mat on the host.
//
// Thread layout: 32×8 blocks. Each warp covers a contiguous row segment,
// improving row-major stores while keeping 256 threads per block.

#include "map_kernel.cuh"
#include "fx64.cuh"

#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)  do {                                                \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess)                                                    \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e) + \
            " at " __FILE__ ":" + std::to_string(__LINE__));                  \
} while(0)

namespace fsd_cuda {

// ---- Device colormap helpers (pixel-exact match with colormap.hpp) ----

__device__ inline int d_clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__device__ inline float d_cos_color(float n, float freq) {
    constexpr float PI = 3.14159265f;
    return 128.0f - 128.0f * cosf(freq * n * PI);
}

__device__ inline void d_hsv_to_rgb(float h, float s, float v,
                                     int& r, int& g, int& b) {
    const float c  = v * s;
    const float hh = h / 60.0f;
    const float x  = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    if      (hh < 1.0f) { rr = c; gg = x; }
    else if (hh < 2.0f) { rr = x; gg = c; }
    else if (hh < 3.0f) { gg = c; bb = x; }
    else if (hh < 4.0f) { gg = x; bb = c; }
    else if (hh < 5.0f) { rr = x; bb = c; }
    else                { rr = c; bb = x; }

    const float m = v - c;
    r = d_clamp255(static_cast<int>((rr + m) * 255.0f));
    g = d_clamp255(static_cast<int>((gg + m) * 255.0f));
    b = d_clamp255(static_cast<int>((bb + m) * 255.0f));
}

// Colorize one escaped pixel.  Matches colormap.hpp pixel-exact.
// colormap_id: 0=ClassicCos, 1=Mod17, 2=HsvWheel, 3=Tri765, 4=Grayscale
// Writes BGR into px[0..2].
__device__ inline void colorize_escape_bgr(int iter, int max_iter,
                                            int colormap_id, uint8_t* px) {
    if (iter >= max_iter) {
        px[0] = px[1] = px[2] = 255;   // interior: white
        return;
    }
    // n matches CPU: (iter+1)/(max_iter+2)
    const float n = (static_cast<float>(iter) + 1.0f) /
                    (static_cast<float>(max_iter) + 2.0f);

    switch (colormap_id) {
        case 1: {  // Mod17
            px[2] = static_cast<uint8_t>(d_clamp255(iter % 256));
            px[1] = static_cast<uint8_t>(d_clamp255(iter / 256));
            px[0] = static_cast<uint8_t>(d_clamp255((iter % 17) * 17));
            return;
        }
        case 2: {  // HsvWheel
            const float h = fmodf(static_cast<float>(iter), 1440.0f) / 4.0f;
            int r = 0, g = 0, b = 0;
            d_hsv_to_rgb(h, 1.0f, 1.0f, r, g, b);
            px[2] = static_cast<uint8_t>(r);
            px[1] = static_cast<uint8_t>(g);
            px[0] = static_cast<uint8_t>(b);
            return;
        }
        case 3: {  // Tri765
            const int m    = iter % 765;
            const int band = m / 255;
            const int d    = m % 255;
            int rr = 255, gg = 255, bb = 255;
            if      (band == 0) { rr = 255 - d; gg = d;       bb = 255;     }
            else if (band == 1) { rr = d;       gg = 255;     bb = 255 - d; }
            else                { rr = 255;     gg = 255 - d; bb = d;       }
            px[2] = static_cast<uint8_t>(d_clamp255(rr));
            px[1] = static_cast<uint8_t>(d_clamp255(gg));
            px[0] = static_cast<uint8_t>(d_clamp255(bb));
            return;
        }
        case 4: {  // Grayscale
            const uint8_t v = static_cast<uint8_t>(d_clamp255(static_cast<int>(n * 255.0f)));
            px[0] = px[1] = px[2] = v;
            return;
        }
        case 5: {  // HsRainbow — for escape, use iter as index cyclically in [0,1785]
            const int idx = iter % 1786;
            int a0 = idx, a1 = 0, a2 = 0;
            if      (255 < a0 && a0 < 510)  { a1 = a0 - 255; a0 = 510 - a0; }
            else if (509 < a0 && a0 < 765)  { a1 = 255; a0 = a0 - 510; }
            else if (764 < a0 && a0 < 1020) { a2 = a0 - 765; a1 = 1020 - a0; a0 = a1; }
            else if (1019 < a0 && a0 < 1275){ a2 = 255; a0 = a0 - 1020; }
            else if (1274 < a0 && a0 < 1530){ a2 = 255; a1 = a0 - 1275; a0 = 1530 - a0; }
            else if (a0 > 1529)              { a2 = 255; a1 = 255; a0 = a0 - 1530; }
            px[2] = static_cast<uint8_t>(d_clamp255(a1));  // R = a1
            px[1] = static_cast<uint8_t>(d_clamp255(a2));  // G = a2
            px[0] = static_cast<uint8_t>(d_clamp255(a0));  // B = a0
            return;
        }
        default:  // 0 = ClassicCos (and any unknown id)
            px[2] = static_cast<uint8_t>(d_clamp255(static_cast<int>(d_cos_color(n,  53.0f))));
            px[1] = static_cast<uint8_t>(d_clamp255(static_cast<int>(d_cos_color(n,  27.0f))));
            px[0] = static_cast<uint8_t>(d_clamp255(static_cast<int>(d_cos_color(n, 139.0f))));
            return;
    }
}

// Colorize a field value in [0,1] using ClassicCos-style formula (colormap_id ignored
// for now; all field metrics use the smooth cosine gradient for visual clarity).
__device__ inline void colorize_field_bgr(double v01, int colormap_id, uint8_t* px) {
    const float n = (float)v01;
    px[2] = (uint8_t)d_clamp255((int)d_cos_color(n, 53.0f));
    px[1] = (uint8_t)d_clamp255((int)d_cos_color(n, 27.0f));
    px[0] = (uint8_t)d_clamp255((int)d_cos_color(n, 139.0f));
}

// ---- fp64 variant step functions ----
// Each __device__ inline function takes (zre, zim, cre, cim) and writes
// (new_zre, new_zim) in-place via reference returns packed as doubles.
// To keep call sites uniform: step takes zre,zim by reference and cre,cim by value.

__device__ inline void step_mandelbrot(double& zre, double& zim, double cre, double cim) {
    const double new_re = zre * zre - zim * zim + cre;
    zim = 2.0 * zre * zim + cim;
    zre = new_re;
}

__device__ inline void step_tri(double& zre, double& zim, double cre, double cim) {
    // conj(z²) + c: negate imaginary part of z²
    const double new_re = zre * zre - zim * zim + cre;
    zim = -(2.0 * zre * zim) + cim;
    zre = new_re;
}

__device__ inline void step_boat(double& zre, double& zim, double cre, double cim) {
    // (|re| + |im|·i)² + c
    const double wre = fabs(zre);
    const double wim = fabs(zim);
    const double new_re = wre * wre - wim * wim + cre;
    zim = 2.0 * wre * wim + cim;
    zre = new_re;
}

__device__ inline void step_duck(double& zre, double& zim, double cre, double cim) {
    // (re + |im|·i)² + c
    const double wim = fabs(zim);
    const double new_re = zre * zre - wim * wim + cre;
    zim = 2.0 * zre * wim + cim;
    zre = new_re;
}

__device__ inline void step_bell(double& zre, double& zim, double cre, double cim) {
    // (|re| - im·i)² + c
    const double wre = fabs(zre);
    const double wim = -zim;
    const double new_re = wre * wre - wim * wim + cre;
    zim = 2.0 * wre * wim + cim;
    zre = new_re;
}

__device__ inline void step_fish(double& zre, double& zim, double cre, double cim) {
    // z² then |re(z²)| + im(z²)·i + c
    const double sq_re = zre * zre - zim * zim;
    const double sq_im = 2.0 * zre * zim;
    zre = fabs(sq_re) + cre;
    zim = sq_im + cim;
}

__device__ inline void step_vase(double& zre, double& zim, double cre, double cim) {
    // z² then |re(z²)| - im(z²)·i + c
    const double sq_re = zre * zre - zim * zim;
    const double sq_im = 2.0 * zre * zim;
    zre = fabs(sq_re) + cre;
    zim = -sq_im + cim;
}

__device__ inline void step_bird(double& zre, double& zim, double cre, double cim) {
    // z² then |re(z²)| + |im(z²)|·i + c
    const double sq_re = zre * zre - zim * zim;
    const double sq_im = 2.0 * zre * zim;
    zre = fabs(sq_re) + cre;
    zim = fabs(sq_im) + cim;
}

__device__ inline void step_mask(double& zre, double& zim, double cre, double cim) {
    // z ← (re + |im|·i), z²=(re+|im|i)², then |re(z²)| + im(z²)·i + c
    const double wim = fabs(zim);
    const double sq_re = zre * zre - wim * wim;
    const double sq_im = 2.0 * zre * wim;
    zre = fabs(sq_re) + cre;
    zim = sq_im + cim;
}

__device__ inline void step_ship(double& zre, double& zim, double cre, double cim) {
    // z ← (|re| + im·i), z²=(|re|+im·i)², then |re(z²)| - im(z²)·i + c
    const double wre = fabs(zre);
    const double sq_re = wre * wre - zim * zim;
    const double sq_im = 2.0 * wre * zim;
    zre = fabs(sq_re) + cre;
    zim = -sq_im + cim;
}

template <int VariantId>
__device__ inline void step_fp64_variant(double& zre, double& zim, double cre, double cim) {
    if constexpr (VariantId == 1)      step_tri(zre, zim, cre, cim);
    else if constexpr (VariantId == 2) step_boat(zre, zim, cre, cim);
    else if constexpr (VariantId == 3) step_duck(zre, zim, cre, cim);
    else if constexpr (VariantId == 4) step_bell(zre, zim, cre, cim);
    else if constexpr (VariantId == 5) step_fish(zre, zim, cre, cim);
    else if constexpr (VariantId == 6) step_vase(zre, zim, cre, cim);
    else if constexpr (VariantId == 7) step_bird(zre, zim, cre, cim);
    else if constexpr (VariantId == 8) step_mask(zre, zim, cre, cim);
    else if constexpr (VariantId == 9) step_ship(zre, zim, cre, cim);
    else                               step_mandelbrot(zre, zim, cre, cim);
}

// ---- fp64 kernel ----

template <int VariantId, int MetricId>
__global__ void fractal_fp64(
    double center_re, double center_im, double scale,
    int W, int H, int max_iter, double bail2, int colormap_id,
    bool julia, double julia_re_p, double julia_im_p,
    uint8_t* __restrict__ out
) {
    const int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (px_x >= W || px_y >= H) return;

    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double span_im = scale;
    const double span_re = scale * aspect;
    const double re = (center_re - span_re * 0.5) + (static_cast<double>(px_x) + 0.5) / W * span_re;
    const double im = (center_im + span_im * 0.5) - (static_cast<double>(px_y) + 0.5) / H * span_im;

    // Initialise z and c based on julia flag.
    double zre, zim, cre, cim;
    if (julia) {
        zre = re;   zim = im;
        cre = julia_re_p; cim = julia_im_p;
    } else {
        zre = 0.0;  zim = 0.0;
        cre = re;   cim = im;
    }

    constexpr bool track_min = (MetricId == 1 || MetricId == 3);
    constexpr bool track_max = (MetricId == 2 || MetricId == 3);
    double mn = INFINITY;
    double mx = 0.0;
    if constexpr (track_min || track_max) {
        const double init_abs2 = zre * zre + zim * zim;
        mn = init_abs2;
        mx = init_abs2;
    }

    // Apply step THEN check — matches escape_time.hpp CPU convention so that
    // r.iter is identical on both paths (both return i when z_{i+1} escapes).
    int i = 0;
    for (; i < max_iter; i++) {
        step_fp64_variant<VariantId>(zre, zim, cre, cim);
        const bool finite_z = isfinite(zre) && isfinite(zim);
        const double abs2 = finite_z ? (zre * zre + zim * zim) : INFINITY;
        if constexpr (track_min) {
            if (abs2 < mn) mn = abs2;
        }
        if constexpr (track_max) {
            if (abs2 > mx) mx = abs2;
        }
        if (!finite_z || abs2 > bail2) break;
    }

    uint8_t* px = out + (static_cast<size_t>(px_y) * W + px_x) * 3;

    if constexpr (MetricId == 1) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, sqrt(mn) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else if constexpr (MetricId == 2) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, sqrt(mx) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else if constexpr (MetricId == 3) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, 0.5 * (sqrt(mn) + sqrt(mx)) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else {
        colorize_escape_bgr(i, max_iter, colormap_id, px);
    }
}

// ---- Fx64 helpers ----

// Absolute value for Fx64: negate raw if negative.
__device__ inline Fx64 fx64_abs(Fx64 x) {
    return Fx64{x.raw < 0 ? -x.raw : x.raw};
}

// ---- fx64 variant step functions ----
// Mirror of the fp64 steps, but using Fx64 fixed-point arithmetic.

__device__ inline void step_mandelbrot_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    const Fx64 new_zre = zre.sqr() - zim.sqr() + cre;
    zim = (zre * zim) + (zre * zim) + cim;
    zre = new_zre;
}

__device__ inline void step_tri_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // conj(z²) + c: negate imaginary part of z²
    const Fx64 new_zre = zre.sqr() - zim.sqr() + cre;
    // 2*zre*zim negated, then +cim
    const Fx64 two_re_im = (zre * zim) + (zre * zim);
    zim = cim - two_re_im;
    zre = new_zre;
}

__device__ inline void step_boat_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // (|re| + |im|·i)² + c
    const Fx64 wre = fx64_abs(zre);
    const Fx64 wim = fx64_abs(zim);
    const Fx64 new_zre = wre.sqr() - wim.sqr() + cre;
    zim = (wre * wim) + (wre * wim) + cim;
    zre = new_zre;
}

__device__ inline void step_duck_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // (re + |im|·i)² + c
    const Fx64 wim = fx64_abs(zim);
    const Fx64 new_zre = zre.sqr() - wim.sqr() + cre;
    zim = (zre * wim) + (zre * wim) + cim;
    zre = new_zre;
}

__device__ inline void step_bell_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // (|re| - im·i)² + c → wre=|zre|, wim=-zim
    const Fx64 wre = fx64_abs(zre);
    const Fx64 wim = -zim;
    const Fx64 new_zre = wre.sqr() - wim.sqr() + cre;
    zim = (wre * wim) + (wre * wim) + cim;
    zre = new_zre;
}

__device__ inline void step_fish_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // z² then |re(z²)| + im(z²)·i + c
    const Fx64 sq_re = zre.sqr() - zim.sqr();
    const Fx64 sq_im = (zre * zim) + (zre * zim);
    zre = fx64_abs(sq_re) + cre;
    zim = sq_im + cim;
}

__device__ inline void step_vase_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // z² then |re(z²)| - im(z²)·i + c
    const Fx64 sq_re = zre.sqr() - zim.sqr();
    const Fx64 sq_im = (zre * zim) + (zre * zim);
    zre = fx64_abs(sq_re) + cre;
    zim = cim - sq_im;
}

__device__ inline void step_bird_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // z² then |re(z²)| + |im(z²)|·i + c
    const Fx64 sq_re = zre.sqr() - zim.sqr();
    const Fx64 sq_im = (zre * zim) + (zre * zim);
    zre = fx64_abs(sq_re) + cre;
    zim = fx64_abs(sq_im) + cim;
}

__device__ inline void step_mask_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // z ← (re + |im|·i), z²=(re+|im|i)², then |re(z²)| + im(z²)·i + c
    const Fx64 wim = fx64_abs(zim);
    const Fx64 sq_re = zre.sqr() - wim.sqr();
    const Fx64 sq_im = (zre * wim) + (zre * wim);
    zre = fx64_abs(sq_re) + cre;
    zim = sq_im + cim;
}

__device__ inline void step_ship_fx(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    // z ← (|re| + im·i), z²=(|re|+im·i)², then |re(z²)| - im(z²)·i + c
    const Fx64 wre = fx64_abs(zre);
    const Fx64 sq_re = wre.sqr() - zim.sqr();
    const Fx64 sq_im = (wre * zim) + (wre * zim);
    zre = fx64_abs(sq_re) + cre;
    zim = cim - sq_im;
}

template <int VariantId>
__device__ inline void step_fx64_variant(Fx64& zre, Fx64& zim, const Fx64& cre, const Fx64& cim) {
    if constexpr (VariantId == 1)      step_tri_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 2) step_boat_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 3) step_duck_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 4) step_bell_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 5) step_fish_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 6) step_vase_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 7) step_bird_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 8) step_mask_fx(zre, zim, cre, cim);
    else if constexpr (VariantId == 9) step_ship_fx(zre, zim, cre, cim);
    else                               step_mandelbrot_fx(zre, zim, cre, cim);
}

// ---- fx64 kernel ----

template <int VariantId, int MetricId>
__global__ void fractal_fx64(
    double center_re_d, double center_im_d, double scale_d,
    int W, int H, int max_iter, double bail2, int colormap_id,
    bool julia, double julia_re_p, double julia_im_p,
    uint8_t* __restrict__ out
) {
    const int px_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int px_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (px_x >= W || px_y >= H) return;

    const double aspect  = static_cast<double>(W) / static_cast<double>(H);
    const double span_im = scale_d;
    const double span_re = scale_d * aspect;
    const double re_d = (center_re_d - span_re * 0.5) + (static_cast<double>(px_x) + 0.5) / W * span_re;
    const double im_d = (center_im_d + span_im * 0.5) - (static_cast<double>(px_y) + 0.5) / H * span_im;

    // Initialise z and c in Fx64 based on julia flag.
    Fx64 zre, zim, cre, cim;
    if (julia) {
        zre = Fx64::from_double(re_d);
        zim = Fx64::from_double(im_d);
        cre = Fx64::from_double(julia_re_p);
        cim = Fx64::from_double(julia_im_p);
    } else {
        zre = {0LL};
        zim = {0LL};
        cre = Fx64::from_double(re_d);
        cim = Fx64::from_double(im_d);
    }

    constexpr bool track_min = (MetricId == 1 || MetricId == 3);
    constexpr bool track_max = (MetricId == 2 || MetricId == 3);
    double mn = INFINITY;
    double mx = 0.0;
    if constexpr (track_min || track_max) {
        const double fre0 = zre.to_double();
        const double fim0 = zim.to_double();
        const double init_abs2 = fre0 * fre0 + fim0 * fim0;
        mn = init_abs2;
        mx = init_abs2;
    }

    // Apply step THEN check — matches escape_time.hpp CPU convention.
    int i = 0;
    for (; i < max_iter; i++) {
        step_fx64_variant<VariantId>(zre, zim, cre, cim);

        // Escape check in fp64 (avoids fixed-point overflow for large |z|)
        const double fre = zre.to_double();
        const double fim = zim.to_double();
        const bool finite_z = isfinite(fre) && isfinite(fim);
        const double abs2 = finite_z ? (fre * fre + fim * fim) : INFINITY;
        if constexpr (track_min) {
            if (abs2 < mn) mn = abs2;
        }
        if constexpr (track_max) {
            if (abs2 > mx) mx = abs2;
        }
        if (!finite_z || abs2 > bail2) break;
    }

    uint8_t* px = out + (static_cast<size_t>(px_y) * W + px_x) * 3;

    if constexpr (MetricId == 1) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, sqrt(mn) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else if constexpr (MetricId == 2) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, sqrt(mx) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else if constexpr (MetricId == 3) {
        const double bailout = sqrt(bail2);
        const double v01 = fmin(1.0, 0.5 * (sqrt(mn) + sqrt(mx)) / bailout);
        colorize_field_bgr(v01, colormap_id, px);
    } else {
        colorize_escape_bgr(i, max_iter, colormap_id, px);
    }
}

// ---- Device output buffer (cached across calls of same size) ----

struct DevBuf {
    int W = 0, H = 0;
    uint8_t* d = nullptr;

    void ensure(int w, int h) {
        if (w == W && h == H) return;
        if (d) { cudaFree(d); d = nullptr; }
        CUDA_CHECK(cudaMalloc(&d, static_cast<size_t>(w) * h * 3));
        W = w; H = h;
    }
    void release() { if (d) { cudaFree(d); d = nullptr; W = H = 0; } }
};

static std::mutex  g_cuda_mutex;
static DevBuf      g_devbuf;

// ---- Public API ----

bool cuda_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <int MetricId>
static void launch_fp64_metric(const CudaMapParams& p, dim3 grid, dim3 block, double bail2, uint8_t* out) {
#define FSD_LAUNCH_FP64(VID) \
    fractal_fp64<VID, MetricId><<<grid, block>>>( \
        p.center_re, p.center_im, p.scale, \
        p.width, p.height, p.iterations, bail2, p.colormap_id, \
        p.julia, p.julia_re, p.julia_im, out)

    switch (p.variant_id) {
        case 1:  FSD_LAUNCH_FP64(1); break;
        case 2:  FSD_LAUNCH_FP64(2); break;
        case 3:  FSD_LAUNCH_FP64(3); break;
        case 4:  FSD_LAUNCH_FP64(4); break;
        case 5:  FSD_LAUNCH_FP64(5); break;
        case 6:  FSD_LAUNCH_FP64(6); break;
        case 7:  FSD_LAUNCH_FP64(7); break;
        case 8:  FSD_LAUNCH_FP64(8); break;
        case 9:  FSD_LAUNCH_FP64(9); break;
        default: FSD_LAUNCH_FP64(0); break;
    }
#undef FSD_LAUNCH_FP64
}

static void launch_fp64(const CudaMapParams& p, dim3 grid, dim3 block, double bail2, uint8_t* out) {
    switch (p.metric_id) {
        case 1:  launch_fp64_metric<1>(p, grid, block, bail2, out); break;
        case 2:  launch_fp64_metric<2>(p, grid, block, bail2, out); break;
        case 3:  launch_fp64_metric<3>(p, grid, block, bail2, out); break;
        default: launch_fp64_metric<0>(p, grid, block, bail2, out); break;
    }
}

template <int MetricId>
static void launch_fx64_metric(const CudaMapParams& p, dim3 grid, dim3 block, double bail2, uint8_t* out) {
#define FSD_LAUNCH_FX64(VID) \
    fractal_fx64<VID, MetricId><<<grid, block>>>( \
        p.center_re, p.center_im, p.scale, \
        p.width, p.height, p.iterations, bail2, p.colormap_id, \
        p.julia, p.julia_re, p.julia_im, out)

    switch (p.variant_id) {
        case 1:  FSD_LAUNCH_FX64(1); break;
        case 2:  FSD_LAUNCH_FX64(2); break;
        case 3:  FSD_LAUNCH_FX64(3); break;
        case 4:  FSD_LAUNCH_FX64(4); break;
        case 5:  FSD_LAUNCH_FX64(5); break;
        case 6:  FSD_LAUNCH_FX64(6); break;
        case 7:  FSD_LAUNCH_FX64(7); break;
        case 8:  FSD_LAUNCH_FX64(8); break;
        case 9:  FSD_LAUNCH_FX64(9); break;
        default: FSD_LAUNCH_FX64(0); break;
    }
#undef FSD_LAUNCH_FX64
}

static void launch_fx64(const CudaMapParams& p, dim3 grid, dim3 block, double bail2, uint8_t* out) {
    switch (p.metric_id) {
        case 1:  launch_fx64_metric<1>(p, grid, block, bail2, out); break;
        case 2:  launch_fx64_metric<2>(p, grid, block, bail2, out); break;
        case 3:  launch_fx64_metric<3>(p, grid, block, bail2, out); break;
        default: launch_fx64_metric<0>(p, grid, block, bail2, out); break;
    }
}

CudaMapStats cuda_render_map(const CudaMapParams& p, cv::Mat& out) {
    if (!cuda_available()) throw std::runtime_error("CUDA not available");

    const bool use_fx = (p.scalar_type == "fx64");

    std::lock_guard<std::mutex> lock(g_cuda_mutex);

    // Ensure device buffer is large enough.
    g_devbuf.ensure(p.width, p.height);

    const dim3 block(32, 8);
    const dim3 grid(
        (p.width + static_cast<int>(block.x) - 1) / static_cast<int>(block.x),
        (p.height + static_cast<int>(block.y) - 1) / static_cast<int>(block.y));
    const double bail2 = p.bailout_sq;

    // Time the kernel with CUDA events.
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    if (use_fx) {
        launch_fx64(p, grid, block, bail2, g_devbuf.d);
    } else {
        launch_fp64(p, grid, block, bail2, g_devbuf.d);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Copy result to host Mat.
    const size_t nbytes = static_cast<size_t>(p.width) * p.height * 3;
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3)
        out.create(p.height, p.width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(out.data, g_devbuf.d, nbytes, cudaMemcpyDeviceToHost));

    CudaMapStats s;
    s.elapsed_ms  = static_cast<double>(ms);
    s.scalar_used = use_fx ? "fx64" : "fp64";
    s.engine_used = "cuda";
    return s;
}

} // namespace fsd_cuda
