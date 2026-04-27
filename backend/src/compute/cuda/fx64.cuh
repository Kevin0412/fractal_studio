// compute/cuda/fx64.cuh
//
// Fixed-point 64-bit helpers for CUDA device code.  The default Fx64 alias is
// Q6.57, with Q4.59 available for component-gated precision mode.

#pragma once

#include <stdint.h>

namespace fsd_cuda {

__host__ __device__ inline uint64_t abs_i64_to_u64(int64_t x) {
    const uint64_t ux = static_cast<uint64_t>(x);
    return x < 0 ? (~ux + 1ULL) : ux;
}

template <int FRAC>
__device__ inline uint64_t fixed_square_q_sat_raw_cuda(int64_t raw);

template <int FRAC>
struct Fixed64 {
    int64_t raw;

    static constexpr int FRAC_BITS = FRAC;
    static constexpr double SCALE = static_cast<double>(1ULL << FRAC);

    __host__ __device__ static Fixed64 from_double(double x) {
        const double hi = static_cast<double>(INT64_MAX) / SCALE;
        const double lo = static_cast<double>(INT64_MIN) / SCALE;
        if (x >= hi) return {INT64_MAX};
        if (x <= lo) return {INT64_MIN};
        return {static_cast<int64_t>(x * SCALE)};
    }

    __host__ __device__ double to_double() const {
        return static_cast<double>(raw) / SCALE;
    }

    __device__ Fixed64 operator+(Fixed64 b) const { return {raw + b.raw}; }
    __device__ Fixed64 operator-(Fixed64 b) const { return {raw - b.raw}; }
    __device__ Fixed64 operator-() const {
        return {raw == INT64_MIN ? INT64_MAX : -raw};
    }

    __device__ Fixed64 operator*(Fixed64 b) const {
        const uint64_t a_abs = abs_i64_to_u64(raw);
        const uint64_t b_abs = abs_i64_to_u64(b.raw);
        const bool neg = (raw ^ b.raw) < 0;

        const uint64_t lo = a_abs * b_abs;
        const uint64_t hi = __umul64hi(a_abs, b_abs);
        constexpr int SHIFT = 64 - FRAC;
        if (hi >= (1ULL << (FRAC - 1))) {
            return {neg ? INT64_MIN : INT64_MAX};
        }
        const uint64_t mag = (hi << SHIFT) | (lo >> FRAC);
        const int64_t signed_mag = static_cast<int64_t>(mag);
        return {neg ? -signed_mag : signed_mag};
    }

    __device__ Fixed64 sqr() const {
        const uint64_t q = fixed_square_q_sat_raw_cuda<FRAC>(raw);
        return {q > static_cast<uint64_t>(INT64_MAX) ? INT64_MAX : static_cast<int64_t>(q)};
    }

    __device__ bool operator>=(Fixed64 b) const { return raw >= b.raw; }
    __device__ bool operator>(Fixed64 b)  const { return raw >  b.raw; }
};

using FxQ657 = Fixed64<57>;
using FxQ459 = Fixed64<59>;
using Fx64 = FxQ657;

template <int FRAC>
__device__ inline uint64_t fixed_square_q_sat_raw_cuda(int64_t raw) {
    const uint64_t a = abs_i64_to_u64(raw);
    const uint64_t lo = a * a;
    const uint64_t hi = __umul64hi(a, a);
    constexpr int SHIFT = 64 - FRAC;
    if (hi >= (1ULL << FRAC)) return UINT64_MAX;
    return (hi << SHIFT) | (lo >> FRAC);
}

template <int FRAC>
__device__ inline uint64_t fixed_mag2_q_sat_cuda(int64_t re_raw, int64_t im_raw) {
    const uint64_t re2 = fixed_square_q_sat_raw_cuda<FRAC>(re_raw);
    const uint64_t im2 = fixed_square_q_sat_raw_cuda<FRAC>(im_raw);
    const uint64_t sum = re2 + im2;
    if (sum < re2) return UINT64_MAX;
    return sum;
}

template <int FRAC>
__device__ inline bool fixed_component_escaped_q_cuda(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout_raw
) {
    (void)FRAC;
    return abs_i64_to_u64(re_raw) > bailout_raw ||
           abs_i64_to_u64(im_raw) > bailout_raw;
}

template <int FRAC>
__device__ inline bool fixed_escaped_q_cuda(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout2_raw
) {
    return fixed_mag2_q_sat_cuda<FRAC>(re_raw, im_raw) > bailout2_raw;
}

__device__ inline uint64_t fx64_square_q57_sat_raw(int64_t raw) {
    return fixed_square_q_sat_raw_cuda<57>(raw);
}

__device__ inline uint64_t fx64_mag2_q57_sat(int64_t re_raw, int64_t im_raw) {
    return fixed_mag2_q_sat_cuda<57>(re_raw, im_raw);
}

__device__ inline bool fx64_component_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout_raw
) {
    return fixed_component_escaped_q_cuda<57>(re_raw, im_raw, bailout_raw);
}

__device__ inline bool fx64_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout2_q57
) {
    return fixed_escaped_q_cuda<57>(re_raw, im_raw, bailout2_q57);
}

} // namespace fsd_cuda
