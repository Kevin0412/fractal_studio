// compute/cuda/fx64.cuh
//
// Fixed-point Fx64 for CUDA device code.
// Same 1s·6i·57f layout as compute/scalar/fx64.hpp but uses __mul64hi /
// __umul64hi instead of __int128 (CUDA doesn't support __int128 portably).
//
// mul via schoolbook:
//   a * b = (a_hi * b_hi) << 64 + (a_hi * b_lo + a_lo * b_hi) << 32 + a_lo * b_lo
// We only need the middle 64 bits after >> 57, so:
//   full high 64 bits  = __mul64hi(a, b)
//   full low  64 bits  = a * b  (truncated C mul)
//   result = (high << (64-57)) | (low >> 57)
//          = (high << 7)       | (low >> 57)
// For signed inputs: sign = sign_a ^ sign_b; use |a|, |b| for the mul,
// then negate result if sign is negative.

#pragma once

#include <stdint.h>

namespace fsd_cuda {

__host__ __device__ inline uint64_t abs_i64_to_u64(int64_t x) {
    const uint64_t ux = static_cast<uint64_t>(x);
    return x < 0 ? (~ux + 1ULL) : ux;
}

struct Fx64 {
    int64_t raw;

    static constexpr int   FRAC  = 57;
    static constexpr double SCALE = 144115188075855872.0;  // 2^57

    __host__ __device__ static Fx64 from_double(double x) {
        if (x >=  64.0) return {INT64_MAX};
        if (x <= -64.0) return {INT64_MIN};
        return {static_cast<int64_t>(x * SCALE)};
    }

    __host__ __device__ double to_double() const {
        return static_cast<double>(raw) / SCALE;
    }

    __device__ Fx64 operator+(Fx64 b) const { return {raw + b.raw}; }
    __device__ Fx64 operator-(Fx64 b) const { return {raw - b.raw}; }
    __device__ Fx64 operator-()        const {
        return {raw == INT64_MIN ? INT64_MAX : -raw};
    }

    __device__ Fx64 operator*(Fx64 b) const {
        // Signed schoolbook multiplication.
        int64_t a = raw, bv = b.raw;
        uint64_t sa = abs_i64_to_u64(a);
        uint64_t sb = abs_i64_to_u64(bv);
        bool neg = (a ^ bv) < 0;

        uint64_t lo = sa * sb;
        uint64_t hi = __umul64hi(sa, sb);

        // result = full128 >> 57  (high << 7) | (low >> 57)
        if (hi >= (1ULL << 56)) {
            return {neg ? INT64_MIN : INT64_MAX};
        }
        uint64_t res = (hi << 7) | (lo >> 57);
        int64_t signed_res = static_cast<int64_t>(res);
        return {neg ? -signed_res : signed_res};
    }

    __device__ Fx64 sqr() const {
        uint64_t a = abs_i64_to_u64(raw);
        uint64_t lo = a * a;
        uint64_t hi = __umul64hi(a, a);
        if (hi >= (1ULL << 56)) return {INT64_MAX};
        uint64_t res = (hi << 7) | (lo >> 57);
        return {static_cast<int64_t>(res)};
    }

    __device__ bool operator>=(Fx64 b) const { return raw >= b.raw; }
    __device__ bool operator>(Fx64 b)  const { return raw >  b.raw; }
};

__device__ inline uint64_t fx64_square_q57_sat_raw(int64_t raw) {
    const uint64_t a = abs_i64_to_u64(raw);
    const uint64_t lo = a * a;
    const uint64_t hi = __umul64hi(a, a);
    if (hi >= (1ULL << 57)) return UINT64_MAX;
    return (hi << 7) | (lo >> 57);
}

__device__ inline uint64_t fx64_mag2_q57_sat(int64_t re_raw, int64_t im_raw) {
    const uint64_t re2 = fx64_square_q57_sat_raw(re_raw);
    const uint64_t im2 = fx64_square_q57_sat_raw(im_raw);
    const uint64_t sum = re2 + im2;
    if (sum < re2) return UINT64_MAX;
    return sum;
}

__device__ inline bool fx64_component_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout_raw
) {
    return abs_i64_to_u64(re_raw) > bailout_raw ||
           abs_i64_to_u64(im_raw) > bailout_raw;
}

__device__ inline bool fx64_escaped_q57(
    int64_t re_raw,
    int64_t im_raw,
    uint64_t bailout2_q57
) {
    return fx64_mag2_q57_sat(re_raw, im_raw) > bailout2_q57;
}

} // namespace fsd_cuda
