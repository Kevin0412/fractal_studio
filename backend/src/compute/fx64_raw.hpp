// compute/fx64_raw.hpp
//
// Host-side Q6.57 raw helpers.  The fx64 render paths use these helpers to keep
// viewport generation, escape checks, and field metrics in fixed-point integer
// space; doubles are only used at API boundaries and final output conversion.

#pragma once

#include "scalar/fx64.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace fsd::compute {

struct Fx64ViewportRaw {
    int64_t first_re_raw = 0;
    int64_t first_im_raw = 0;
    int64_t step_re_raw = 0;
    int64_t step_im_raw = 0;
    int64_t julia_re_raw = 0;
    int64_t julia_im_raw = 0;
    uint64_t bailout_raw = 0;
    uint64_t bailout2_q57 = 0;
};

inline uint64_t abs_i64_to_u64(int64_t x) noexcept {
    const uint64_t ux = static_cast<uint64_t>(x);
    return x < 0 ? (~ux + 1ULL) : ux;
}

inline uint64_t fx64_square_q57_sat_raw_cpu(int64_t raw) noexcept {
    const uint64_t a = abs_i64_to_u64(raw);
    const unsigned __int128 product =
        static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(a);
    const unsigned __int128 shifted = product >> Fx64::FRAC;
    if (shifted > std::numeric_limits<uint64_t>::max()) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(shifted);
}

inline uint64_t fx64_square_q57_sat_raw(int64_t raw) noexcept {
    return fx64_square_q57_sat_raw_cpu(raw);
}

inline uint64_t fx64_mag2_q57_sat_cpu(int64_t re_raw, int64_t im_raw) noexcept {
    const uint64_t re2 = fx64_square_q57_sat_raw_cpu(re_raw);
    const uint64_t im2 = fx64_square_q57_sat_raw_cpu(im_raw);
    const uint64_t sum = re2 + im2;
    if (sum < re2) return std::numeric_limits<uint64_t>::max();
    return sum;
}

inline uint64_t fx64_mag2_q57_sat(int64_t re_raw, int64_t im_raw) noexcept {
    return fx64_mag2_q57_sat_cpu(re_raw, im_raw);
}

inline bool fx64_component_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout_raw
) noexcept {
    return abs_i64_to_u64(re_raw) > bailout_raw ||
           abs_i64_to_u64(im_raw) > bailout_raw;
}

inline bool fx64_escaped_q57_cpu(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout2_q57
) noexcept {
    return fx64_mag2_q57_sat_cpu(re_raw, im_raw) > bailout2_q57;
}

inline bool fx64_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout2_q57
) noexcept {
    return fx64_escaped_q57_cpu(re_raw, im_raw, bailout2_q57);
}

inline double fx64_mag2_q57_to_double(uint64_t mag2_q57) noexcept {
    return static_cast<double>(mag2_q57) / Fx64::SCALE;
}

inline double fx64_mag2_q57_to_abs(uint64_t mag2_q57) noexcept {
    return std::sqrt(fx64_mag2_q57_to_double(mag2_q57));
}

inline int64_t fx64_round_to_raw_sat(double x) noexcept {
    if (!std::isfinite(x)) return x < 0.0 ? std::numeric_limits<int64_t>::min()
                                          : std::numeric_limits<int64_t>::max();
    const long double scaled = static_cast<long double>(x) *
        static_cast<long double>(UINT64_C(1) << Fx64::FRAC);
    if (scaled >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    if (scaled <= static_cast<long double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(std::llround(scaled));
}

inline uint64_t fx64_round_to_uraw_sat(double x) noexcept {
    if (!std::isfinite(x) || x <= 0.0) return 0;
    const long double scaled = static_cast<long double>(x) *
        static_cast<long double>(UINT64_C(1) << Fx64::FRAC);
    if (scaled >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(std::floor(scaled + 0.5L));
}

inline int64_t fx64_saturate_i128(__int128 value) noexcept {
    if (value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(value);
}

inline Fx64ViewportRaw make_fx64_viewport_raw(
    double center_re,
    double center_im,
    double scale,
    int width,
    int height,
    double julia_re,
    double julia_im,
    double bailout,
    double bailout_sq
) noexcept {
    Fx64ViewportRaw v;
    const int64_t center_re_raw = fx64_round_to_raw_sat(center_re);
    const int64_t center_im_raw = fx64_round_to_raw_sat(center_im);
    const int64_t scale_raw = fx64_round_to_raw_sat(scale);

    const __int128 span_im_raw = static_cast<__int128>(scale_raw);
    const __int128 span_re_raw =
        height > 0 ? (static_cast<__int128>(scale_raw) * width) / height : 0;
    v.step_re_raw = width > 0
        ? fx64_saturate_i128(span_re_raw / width)
        : 0;
    v.step_im_raw = height > 0
        ? fx64_saturate_i128(span_im_raw / height)
        : 0;

    v.first_re_raw = fx64_saturate_i128(
        static_cast<__int128>(center_re_raw) - span_re_raw / 2 + v.step_re_raw / 2);
    v.first_im_raw = fx64_saturate_i128(
        static_cast<__int128>(center_im_raw) + span_im_raw / 2 - v.step_im_raw / 2);
    v.julia_re_raw = fx64_round_to_raw_sat(julia_re);
    v.julia_im_raw = fx64_round_to_raw_sat(julia_im);
    v.bailout_raw = fx64_round_to_uraw_sat(bailout);
    v.bailout2_q57 = fx64_round_to_uraw_sat(bailout_sq);
    return v;
}

inline int64_t fx64_pixel_re_raw(const Fx64ViewportRaw& v, int x) noexcept {
    return fx64_saturate_i128(
        static_cast<__int128>(v.first_re_raw) +
        static_cast<__int128>(x) * v.step_re_raw);
}

inline int64_t fx64_pixel_im_raw(const Fx64ViewportRaw& v, int y) noexcept {
    return fx64_saturate_i128(
        static_cast<__int128>(v.first_im_raw) -
        static_cast<__int128>(y) * v.step_im_raw);
}

} // namespace fsd::compute
