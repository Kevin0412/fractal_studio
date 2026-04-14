// compute/scalar/fx64.hpp
//
// Fixed-point scalar: 1 sign · 6 integer · 57 fractional bits.
//
//   raw = Q × 2^57     →   representable range ≈ ±64
//   precision          =   2^-57 ≈ 6.9e-18   (vs fp64's 2.2e-16 significand)
//
// The Mandelbrot set is bounded by |c| ≤ 2 and escape radius 4, so ±64 is
// plenty. At extreme zoom (scale < 1e-13) fp64 loses ≈ 13 bits of mantissa to
// the integer part of the coordinate — Fx64 retains all 57 fractional bits.
//
// Arithmetic rules:
//   add / sub  → direct int64 add/sub (no overflow in practice — values stay < 64)
//   mul / sqr  → __int128 intermediate: ((__int128)a * b) >> 57
//   cabs       → sqrt via fp64 round-trip (cheap, used only on escape check)
//   from/to    → round toward zero, saturating
//
// This header is self-contained — include it wherever Fx64 kernels are needed.

#pragma once

#include <cmath>
#include <cstdint>

namespace fsd::compute {

struct Fx64 {
    int64_t raw;

    static constexpr int FRAC = 57;
    // 2^57 as a double, used for conversions
    static constexpr double SCALE = static_cast<double>(INT64_C(1) << FRAC);

    // Construction
    constexpr Fx64() noexcept : raw(0) {}
    // Raw-bits constructor (from int64_t): used internally for saturation limits.
    explicit constexpr Fx64(int64_t r) noexcept : raw(r) {}
    // Value constructor (from int): Fx64(2) = 2.0 in fixed-point.
    // This makes `S(2)` work in generic Cx<S> / kernel code.
    explicit constexpr Fx64(int v) noexcept : raw(static_cast<int64_t>(v) << FRAC) {}

    // Explicit conversion to double so static_cast<double>(fx64_val) works in
    // generic kernel code (escape_time.hpp uses scalar_to_double, but having
    // this operator available avoids surprises).
    explicit operator double() const noexcept { return to_double(); }

    // Conversion from double (saturating, round toward zero)
    static Fx64 from_double(double x) noexcept {
        if (x >= 64.0)  return Fx64{ INT64_MAX };
        if (x <= -64.0) return Fx64{ INT64_MIN };
        return Fx64{ static_cast<int64_t>(x * SCALE) };
    }

    double to_double() const noexcept {
        return static_cast<double>(raw) / SCALE;
    }

    // Arithmetic
    Fx64 operator+(Fx64 b) const noexcept { return Fx64{ raw + b.raw }; }
    Fx64 operator-(Fx64 b) const noexcept { return Fx64{ raw - b.raw }; }
    Fx64 operator-()        const noexcept { return Fx64{ -raw }; }

    // Multiply: use __int128 to avoid overflow in the product
    Fx64 operator*(Fx64 b) const noexcept {
        const __int128 p = static_cast<__int128>(raw) * b.raw;
        return Fx64{ static_cast<int64_t>(p >> FRAC) };
    }

    // Square (a == b optimised path)
    Fx64 sqr() const noexcept {
        const __int128 p = static_cast<__int128>(raw) * raw;
        return Fx64{ static_cast<int64_t>(p >> FRAC) };
    }

    // Comparisons
    bool operator>=(Fx64 b) const noexcept { return raw >= b.raw; }
    bool operator<=(Fx64 b) const noexcept { return raw <= b.raw; }
    bool operator>(Fx64 b)  const noexcept { return raw >  b.raw; }
    bool operator<(Fx64 b)  const noexcept { return raw <  b.raw; }
    bool operator==(Fx64 b) const noexcept { return raw == b.raw; }
};

// Scalar overload points for Fx64
inline Fx64 scalar_abs(Fx64 x) noexcept {
    return x.raw >= 0 ? x : Fx64{ -x.raw };
}

inline Fx64 scalar_sqrt(Fx64 x) noexcept {
    // Route through fp64 — only called for cabs in the escape check.
    return Fx64::from_double(std::sqrt(x.to_double()));
}

// Note: scalar_from_double<Fx64>(x) is handled by the primary template in
// scalar.hpp which calls S::from_double(x). No separate overload here.
inline double scalar_to_double(Fx64 x)  noexcept { return x.to_double(); }

} // namespace fsd::compute
