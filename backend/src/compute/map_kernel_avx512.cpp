// compute/map_kernel_avx512.cpp
//
// AVX-512F fractal map renderer — 8 pixels at a time (fp64 or Fx64).
//
// Architecture:
//   zmm regs hold 8 fp64 values. Each "lane" is one pixel. We iterate all 8
//   together until all lanes have escaped or max_iter is reached. Escaped
//   lanes are kept in a bitmask; their zmm values are no longer updated (the
//   mask prevents writing back). This is the "masked iteration" pattern.
//
// Variant support: all 10 variants are supported in the fp64 path via a
//   runtime switch on variant_id *outside* the pixel loop. Each variant has
//   its own tight inner AVX-512 loop. The fx64 (IFMA52) path supports
//   Mandelbrot only (512-bit IFMA for other variants is complex); non-
//   Mandelbrot variants fall through to the scalar OpenMP path.
//
// Julia mode:
//   When p.julia is true, the pixel coordinate is z0 and the constant c is
//   (p.julia_re, p.julia_im). The Mandelbrot setup is the inverse: z0=0,
//   c=pixel. This is handled by swapping the initialisation of vzre/vzim and
//   vcre/vcim after the coordinate build step.
//
// Non-escape metrics:
//   vmn (min |z|²) and vmx (max |z|²) are tracked across all iterations in
//   the AVX-512 loop. After the loop, per-lane mn/mx are extracted and fed to
//   colorize_field_bgr for MinAbs / MaxAbs / Envelope metrics.
//
// Fx64 path:
//   Uses __m512i (8 × int64_t). Multiplication uses IFMA52
//   (_mm512_madd52lo/hi_epu64) for the lower/upper 52-bit partial products,
//   then combines them. The full 57-bit product is reconstructed as:
//     prod = (lo52 | (hi52 << 52)) >> 5   [with sign handling]
//   For correctness, intermediate values are kept as int64_t and only the
//   final shifted result is used. A sign-correction pass handles negative
//   operands by negating before/after (IFMA operates on unsigned).

#include "map_kernel_avx512.hpp"
#include "colormap.hpp"   // colorize_escape_bgr / colorize_field_bgr
#include "variants.hpp"   // Variant enum

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <cstring>

// AVX-512 intrinsics are available if __AVX512F__ is defined.
// We compile this file with -mavx512f via a target_compile_options guard in
// CMakeLists.txt so the intrinsics are always available here.
#if defined(__AVX512F__)
#  include <immintrin.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#  include <cpuid.h>  // __get_cpuid_count
#endif

namespace fsd::compute {

bool avx512_available() noexcept {
#if defined(__AVX512F__) && (defined(__x86_64__) || defined(__i386__))
    // Runtime CPUID check — the executable may run on a non-AVX512 host.
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return static_cast<bool>((ebx >> 16) & 1);
    }
    return false;
#else
    return false;
#endif
}

#if defined(__AVX512F__)

// ---- fp64 AVX-512 kernel — all 10 variants, Julia mode, metrics 0-3 ----

// Helper: normalize min|z|² or max|z|² to [0,1] for field coloring.
// bail2 = bailout².
static inline double norm2_to_01(double v, double bail2) {
    if (!std::isfinite(v) || v <= 0.0) return 0.0;
    return std::min(1.0, std::sqrt(v) / std::sqrt(bail2));
}

static void avx512_fp64_row(
    int y, int W, int H,
    double re_min, double im_max,
    double span_re, double span_im,
    double bail2, int max_iter,
    int variant_id,
    bool julia, double julia_re, double julia_im,
    Metric metric, Colormap cmap,
    uint8_t* row_ptr
) {
    const double im_d = im_max - (static_cast<double>(y) + 0.5) / H * span_im;
    const __m512d vbail2 = _mm512_set1_pd(bail2);
    const __m512d vtwo   = _mm512_set1_pd(2.0);
    const __m512d vzero  = _mm512_setzero_pd();

    // Pixel x stride: 8 lanes at a time.
    for (int x = 0; x < W; x += 8) {
        // Build coordinate values for 8 consecutive pixels.
        double re_arr[8];
        for (int k = 0; k < 8; k++) {
            const int px_x = x + k;
            re_arr[k] = (px_x < W)
                ? re_min + (static_cast<double>(px_x) + 0.5) / W * span_re
                : 1.0e30;  // out-of-bounds lane: escapes immediately
        }
        const __m512d vpx_re = _mm512_loadu_pd(re_arr);
        const __m512d vpx_im = _mm512_set1_pd(im_d);

        // Julia vs Mandelbrot initialisation.
        // Mandelbrot: z0 = 0,   c = pixel coord
        // Julia:      z0 = pixel coord,  c = (julia_re, julia_im)
        __m512d vzre, vzim, vcre, vcim;
        if (julia) {
            vzre = vpx_re;
            vzim = vpx_im;
            vcre = _mm512_set1_pd(julia_re);
            vcim = _mm512_set1_pd(julia_im);
        } else {
            vzre = vzero;
            vzim = vzero;
            vcre = vpx_re;
            vcim = vpx_im;
        }

        __m512i viter  = _mm512_setzero_si512();
        __mmask8 active = 0xFF;

        // Track min/max |z|² for non-escape metrics.
        // Initialise with |z0|².
        __m512d vn2_init = _mm512_add_pd(
            _mm512_mul_pd(vzre, vzre), _mm512_mul_pd(vzim, vzim));
        __m512d vmn = vn2_init;
        __m512d vmx = vn2_init;

        // Select which variant inner loop to run. The switch is *outside* the
        // pixel loop (hoisted per-row) so the branch predictor and the
        // out-of-order engine see a straight-line inner loop with no variant
        // check per iteration.

#define AVX_INNER_LOOP_BEGIN \
        for (int i = 0; i < max_iter && active; i++) {

#define AVX_WRITEBACK_AND_ESCAPE \
            vzre = _mm512_mask_mov_pd(vzre, active, new_re); \
            vzim = _mm512_mask_mov_pd(vzim, active, new_im); \
            const __m512d vn2 = _mm512_add_pd( \
                _mm512_mul_pd(vzre, vzre), _mm512_mul_pd(vzim, vzim)); \
            vmn = _mm512_min_pd(vmn, vn2); \
            vmx = _mm512_max_pd(vmx, vn2); \
            const __mmask8 escaped = _mm512_mask_cmp_pd_mask( \
                active, vn2, vbail2, _CMP_GT_OQ); \
            if (escaped) { \
                const __m512i vi = _mm512_set1_epi64(i); \
                viter = _mm512_mask_mov_epi64(viter, escaped, vi); \
                active &= ~escaped; \
            } \
        }

        switch (variant_id) {

        // ---- 0: Mandelbrot  z' = z² + c -----------------------------------
        case 0: default:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d vzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_sub_pd(vzre2, vzim2), vcre);
            const __m512d new_im = _mm512_add_pd(
                _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), vzim), vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 1: Tri (tricorn)  z' = conj(z²) + c --------------------------
        // new_re = re²-im² + cre  (same as Mandelbrot)
        // new_im = -(2*re*im) + cim
        case 1:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d vzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_sub_pd(vzre2, vzim2), vcre);
            // fnmadd: -(2*re*im) + cim
            const __m512d new_im = _mm512_fnmadd_pd(
                _mm512_mul_pd(vtwo, vzre), vzim, vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 2: Boat  w=(|re|,|im|), z'=w²+c ------------------------------
        case 2:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d wre    = _mm512_abs_pd(vzre);
            const __m512d wim    = _mm512_abs_pd(vzim);
            const __m512d wre2   = _mm512_mul_pd(wre, wre);
            const __m512d wim2   = _mm512_mul_pd(wim, wim);
            const __m512d new_re = _mm512_add_pd(_mm512_sub_pd(wre2, wim2), vcre);
            const __m512d new_im = _mm512_add_pd(
                _mm512_mul_pd(_mm512_mul_pd(vtwo, wre), wim), vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 3: Duck  w=(re,|im|), z'=w²+c --------------------------------
        case 3:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d wim    = _mm512_abs_pd(vzim);
            const __m512d wzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d wim2   = _mm512_mul_pd(wim, wim);
            const __m512d new_re = _mm512_add_pd(_mm512_sub_pd(wzre2, wim2), vcre);
            const __m512d new_im = _mm512_add_pd(
                _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), wim), vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 4: Bell  w=(|re|,-im), z'=w²+c --------------------------------
        case 4:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d wre    = _mm512_abs_pd(vzre);
            const __m512d wim    = _mm512_sub_pd(vzero, vzim);  // -vzim
            const __m512d wre2   = _mm512_mul_pd(wre, wre);
            const __m512d wim2   = _mm512_mul_pd(wim, wim);
            const __m512d new_re = _mm512_add_pd(_mm512_sub_pd(wre2, wim2), vcre);
            const __m512d new_im = _mm512_add_pd(
                _mm512_mul_pd(_mm512_mul_pd(vtwo, wre), wim), vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 5: Fish  sq=z²; new=(|sq.re|, sq.im)+c -----------------------
        case 5:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d vzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d sq_re  = _mm512_sub_pd(vzre2, vzim2);
            const __m512d sq_im  = _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_abs_pd(sq_re), vcre);
            const __m512d new_im = _mm512_add_pd(sq_im, vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 6: Vase  sq=z²; new=(|sq.re|, -sq.im)+c ----------------------
        case 6:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d vzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d sq_re  = _mm512_sub_pd(vzre2, vzim2);
            const __m512d sq_im  = _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_abs_pd(sq_re), vcre);
            // fnmadd: -(sq_im) + vcim
            const __m512d new_im = _mm512_fnmadd_pd(
                _mm512_set1_pd(1.0), sq_im, vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 7: Bird  sq=z²; new=(|sq.re|,|sq.im|)+c ----------------------
        case 7:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d vzre2  = _mm512_mul_pd(vzre, vzre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d sq_re  = _mm512_sub_pd(vzre2, vzim2);
            const __m512d sq_im  = _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_abs_pd(sq_re), vcre);
            const __m512d new_im = _mm512_add_pd(_mm512_abs_pd(sq_im), vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 8: Mask  w=(re,|im|), sq=w²; new=(|sq.re|,sq.im)+c -----------
        case 8:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d wim    = _mm512_abs_pd(vzim);
            const __m512d wre2   = _mm512_mul_pd(vzre, vzre);
            const __m512d wim2   = _mm512_mul_pd(wim, wim);
            const __m512d sq_re  = _mm512_sub_pd(wre2, wim2);
            const __m512d sq_im  = _mm512_mul_pd(_mm512_mul_pd(vtwo, vzre), wim);
            const __m512d new_re = _mm512_add_pd(_mm512_abs_pd(sq_re), vcre);
            const __m512d new_im = _mm512_add_pd(sq_im, vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        // ---- 9: Ship  w=(|re|,im), sq=w²; new=(|sq.re|,-sq.im)+c ----------
        case 9:
        AVX_INNER_LOOP_BEGIN
        {
            const __m512d wre    = _mm512_abs_pd(vzre);
            const __m512d wre2   = _mm512_mul_pd(wre, wre);
            const __m512d vzim2  = _mm512_mul_pd(vzim, vzim);
            const __m512d sq_re  = _mm512_sub_pd(wre2, vzim2);
            const __m512d sq_im  = _mm512_mul_pd(_mm512_mul_pd(vtwo, wre), vzim);
            const __m512d new_re = _mm512_add_pd(_mm512_abs_pd(sq_re), vcre);
            // fnmadd: -(sq_im) + vcim
            const __m512d new_im = _mm512_fnmadd_pd(
                _mm512_set1_pd(1.0), sq_im, vcim);
            AVX_WRITEBACK_AND_ESCAPE
        }
        break;

        } // switch (variant_id)

#undef AVX_INNER_LOOP_BEGIN
#undef AVX_WRITEBACK_AND_ESCAPE

        // Write output pixels (up to 8, clamped to actual W).
        int64_t iters_arr[8];
        _mm512_storeu_si512(iters_arr, viter);
        double mn_arr[8], mx_arr[8];
        _mm512_storeu_pd(mn_arr, vmn);
        _mm512_storeu_pd(mx_arr, vmx);

        for (int k = 0; k < 8 && (x + k) < W; k++) {
            uint8_t* px = row_ptr + 3 * (x + k);
            const bool escaped_k = !((active >> k) & 1);

            if (metric == Metric::Escape) {
                const int it = escaped_k ? static_cast<int>(iters_arr[k]) : max_iter;
                // norm not tracked in AVX-512 path; smooth mode excluded at dispatch.
                colorize_escape_bgr(it, max_iter, cmap, 0.0, false, px[0], px[1], px[2]);
            } else if (metric == Metric::MinAbs) {
                const double v01 = norm2_to_01(mn_arr[k], bail2);
                colorize_field_bgr(v01, cmap, px[0], px[1], px[2]);
            } else if (metric == Metric::MaxAbs) {
                const double v01 = norm2_to_01(mx_arr[k], bail2);
                colorize_field_bgr(v01, cmap, px[0], px[1], px[2]);
            } else {
                // Envelope: combine min+max.
                const double mn_v = norm2_to_01(mn_arr[k], bail2);
                const double mx_v = norm2_to_01(mx_arr[k], bail2);
                colorize_field_bgr(0.5 * (mn_v + mx_v), cmap, px[0], px[1], px[2]);
            }
        }
    }
}

MapStats render_map_avx512_fp64(const MapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    const int W = p.width, H = p.height;
    const double aspect  = static_cast<double>(W) / H;
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout * p.bailout;
    const int variant_id = static_cast<int>(p.variant);

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < H; y++) {
        avx512_fp64_row(
            y, W, H, re_min, im_max, span_re, span_im,
            bail2, p.iterations,
            variant_id,
            p.julia, p.julia_re, p.julia_im,
            p.metric, p.colormap,
            out.ptr<uint8_t>(y)
        );
    }

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = W * H;
    s.scalar_used = "fp64";
    s.engine_used = "avx512";
    return s;
}

// ---- Fx64 AVX-512 kernel (Mandelbrot only, IFMA52) ----
//
// IFMA52: _mm512_madd52lo_epu64 / _mm512_madd52hi_epu64
//   dst[i] += (a[i] * b[i])[51:0]   (lo)
//   dst[i] += (a[i] * b[i])[103:52] (hi)
//
// Our fixed-point uses FRAC=57 bits. For two positive int64_t a,b (raw):
//   product = (a * b) >> 57
//
// Signed treatment: for sign correctness we track signs separately.
// negate_result = sign_a XOR sign_b, then use |a|, |b| for IFMA.
//
// Julia mode: when julia=true, z0 = pixel coord, c = (julia_re, julia_im).

static void avx512_mandelbrot_row_fx64(
    int y, int W, int H,
    double re_min_d, double im_max_d,
    double span_re_d, double span_im_d,
    double bail2, int max_iter,
    bool julia, double julia_re, double julia_im,
    Metric metric, Colormap cmap,
    uint8_t* row_ptr
) {
    [[maybe_unused]] static constexpr int FRAC = 57;

    const double im_d = im_max_d - (static_cast<double>(y) + 0.5) / H * span_im_d;

    // Fixed-point representation of the Julia constant (used when julia=true).
    const int64_t julia_re_raw = static_cast<int64_t>(julia_re * Fx64::SCALE);
    const int64_t julia_im_raw = static_cast<int64_t>(julia_im * Fx64::SCALE);

    for (int x = 0; x < W; x += 8) {
        // Build raw fixed-point c values for 8 pixels.
        int64_t px_re_arr[8], px_im_arr[8];
        for (int k = 0; k < 8; k++) {
            const int px_x = x + k;
            if (px_x < W) {
                const double re_d = re_min_d + (static_cast<double>(px_x) + 0.5) / W * span_re_d;
                px_re_arr[k] = static_cast<int64_t>(re_d * Fx64::SCALE);
            } else {
                px_re_arr[k] = INT64_MAX;  // will escape immediately
            }
            px_im_arr[k] = static_cast<int64_t>(im_d * Fx64::SCALE);
        }

        const __m512i vpx_re = _mm512_loadu_si512(px_re_arr);
        const __m512i vpx_im = _mm512_loadu_si512(px_im_arr);

        // Julia vs Mandelbrot z0/c setup.
        __m512i vcre, vcim, vzre, vzim;
        if (julia) {
            vzre = vpx_re;
            vzim = vpx_im;
            vcre = _mm512_set1_epi64(julia_re_raw);
            vcim = _mm512_set1_epi64(julia_im_raw);
        } else {
            vzre = _mm512_setzero_si512();
            vzim = _mm512_setzero_si512();
            vcre = vpx_re;
            vcim = vpx_im;
        }

        __m512i viter = _mm512_setzero_si512();
        __mmask8 active = 0xFF;

        // Bail² as fixed-point (in fp64 for escape check).
        for (int i = 0; i < max_iter && active; i++) {
            // z' = z² + c
            // re' = re*re - im*im + cre
            // im' = 2*re*im + cim
            //
            // Fixed-point multiplication via IFMA52:
            // For signed operands a, b:
            //   sign_r = sign_a XOR sign_b
            //   |product_raw| = (|a| * |b|) >> FRAC
            //   product = sign_r ? -|product_raw| : |product_raw|

            // Masks for sign bits (arithmetic shift right by 63 gives all-bits sign)
            const __m512i vsign_re = _mm512_srai_epi64(vzre, 63);
            const __m512i vsign_im = _mm512_srai_epi64(vzim, 63);

            // Absolute values.
            // abs(x) = (x XOR sign) - sign  (two's complement trick)
            const __m512i vabs_re = _mm512_sub_epi64(
                _mm512_xor_si512(vzre, vsign_re), vsign_re);
            const __m512i vabs_im = _mm512_sub_epi64(
                _mm512_xor_si512(vzim, vsign_im), vsign_im);

            // re*re (positive × positive)
            __m512i vprod_lo = _mm512_setzero_si512();
            __m512i vprod_hi = _mm512_setzero_si512();
            vprod_lo = _mm512_madd52lo_epu64(vprod_lo, vabs_re, vabs_re);
            vprod_hi = _mm512_madd52hi_epu64(vprod_hi, vabs_re, vabs_re);
            // Reconstruct: full product >> FRAC (= 57)
            // lo has bits [51:0], hi has bits [103:52]
            // full = (vprod_hi << 52) | vprod_lo
            // result = full >> 57 = vprod_hi >> 5 | vprod_lo >> 57
            const __m512i vre2 = _mm512_or_si512(
                _mm512_srli_epi64(vprod_hi, 5),
                _mm512_srli_epi64(vprod_lo, 57)
            );

            // im*im (positive × positive)
            vprod_lo = _mm512_setzero_si512();
            vprod_hi = _mm512_setzero_si512();
            vprod_lo = _mm512_madd52lo_epu64(vprod_lo, vabs_im, vabs_im);
            vprod_hi = _mm512_madd52hi_epu64(vprod_hi, vabs_im, vabs_im);
            const __m512i vim2 = _mm512_or_si512(
                _mm512_srli_epi64(vprod_hi, 5),
                _mm512_srli_epi64(vprod_lo, 57)
            );

            // new_re = re2 - im2 + cre  (all signs already positive at this point)
            const __m512i new_re = _mm512_add_epi64(
                _mm512_sub_epi64(vre2, vim2), vcre);

            // re*im: signs of re and im may differ
            const __m512i vsign_re_im = _mm512_xor_si512(vsign_re, vsign_im);
            vprod_lo = _mm512_setzero_si512();
            vprod_hi = _mm512_setzero_si512();
            vprod_lo = _mm512_madd52lo_epu64(vprod_lo, vabs_re, vabs_im);
            vprod_hi = _mm512_madd52hi_epu64(vprod_hi, vabs_re, vabs_im);
            __m512i vrim = _mm512_or_si512(
                _mm512_srli_epi64(vprod_hi, 5),
                _mm512_srli_epi64(vprod_lo, 57)
            );
            // Apply sign: negate if sign_re XOR sign_im is negative (all-ones)
            vrim = _mm512_sub_epi64(
                _mm512_xor_si512(vrim, vsign_re_im), vsign_re_im);

            // new_im = 2 * re * im + cim
            const __m512i new_im = _mm512_add_epi64(
                _mm512_slli_epi64(vrim, 1), vcim);

            vzre = _mm512_mask_mov_epi64(vzre, active, new_re);
            vzim = _mm512_mask_mov_epi64(vzim, active, new_im);

            // Escape check: convert to fp64 for the comparison
            const __m512d fd_re = _mm512_cvt_roundepi64_pd(vzre,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const __m512d fd_im = _mm512_cvt_roundepi64_pd(vzim,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const __m512d scale_inv = _mm512_set1_pd(1.0 / Fx64::SCALE);
            const __m512d fre = _mm512_mul_pd(fd_re, scale_inv);
            const __m512d fim = _mm512_mul_pd(fd_im, scale_inv);
            const __m512d vn2 = _mm512_fmadd_pd(fim, fim,
                _mm512_mul_pd(fre, fre));
            const __m512d vbail2 = _mm512_set1_pd(bail2);
            const __mmask8 escaped = _mm512_mask_cmp_pd_mask(
                active, vn2, vbail2, _CMP_GT_OQ);

            if (escaped) {
                const __m512i vi = _mm512_set1_epi64(i);
                viter = _mm512_mask_mov_epi64(viter, escaped, vi);
                active &= ~escaped;
            }
        }

        int64_t iters_arr[8];
        _mm512_storeu_si512(iters_arr, viter);
        for (int k = 0; k < 8 && (x + k) < W; k++) {
            uint8_t* px = row_ptr + 3 * (x + k);
            const bool escaped_k = !((active >> k) & 1);
            const int it = escaped_k ? static_cast<int>(iters_arr[k]) : max_iter;
            colorize_escape_bgr(it, max_iter, cmap, 0.0, false, px[0], px[1], px[2]);
        }
    }
}

MapStats render_map_avx512_fx64(const MapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height || out.cols != p.width || out.type() != CV_8UC3) {
        out.create(p.height, p.width, CV_8UC3);
    }

    const int W = p.width, H = p.height;
    const double aspect  = static_cast<double>(W) / H;
    const double span_im = p.scale;
    const double span_re = p.scale * aspect;
    const double re_min  = p.center_re - span_re * 0.5;
    const double im_max  = p.center_im + span_im * 0.5;
    const double bail2   = p.bailout * p.bailout;

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < H; y++) {
        avx512_mandelbrot_row_fx64(
            y, W, H, re_min, im_max, span_re, span_im,
            bail2, p.iterations,
            p.julia, p.julia_re, p.julia_im,
            p.metric, p.colormap,
            out.ptr<uint8_t>(y)
        );
    }

    const auto t1 = std::chrono::steady_clock::now();
    MapStats s;
    s.elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.pixel_count = W * H;
    s.scalar_used = "fx64";
    s.engine_used = "avx512";
    return s;
}

#else  // !__AVX512F__

// Stub implementations when AVX-512 is not available at compile time.
MapStats render_map_avx512_fp64(const MapParams& p, cv::Mat& out) {
    (void)p; (void)out;
    MapStats s; s.engine_used = "openmp_fallback"; return s;
}
MapStats render_map_avx512_fx64(const MapParams& p, cv::Mat& out) {
    (void)p; (void)out;
    MapStats s; s.engine_used = "openmp_fallback"; return s;
}

#endif  // __AVX512F__

} // namespace fsd::compute
