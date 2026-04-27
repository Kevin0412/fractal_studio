// compute/scalar/fx64.hpp
//
// Fixed-point scalar templates for signed 64-bit Q formats.
//
//   raw = Q × 2^FRAC
//
// The default Fx64 alias remains Q6.57 for compatibility.  Q4.59 and Q3.60 are
// available for gated precision modes where the orbit range is constrained
// before the exact radius check.
//
// Arithmetic rules:
//   add / sub  → direct int64 add/sub (no overflow in practice — values stay < 64)
//   mul / sqr  → __int128 intermediate: ((__int128)a * b) >> FRAC
//   cabs       → sqrt via fp64 round-trip for non-hot output/helper paths
//   from/to    → round toward zero, saturating
//
// This header is self-contained — include it wherever fixed-point kernels are
// needed.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace fsd::compute {

template <int FRAC_>
struct Fixed64 {
    static_assert(FRAC_ > 0 && FRAC_ < 63, "Fixed64 requires 0 < FRAC < 63.");

    int64_t raw;

    static constexpr int FRAC = FRAC_;
    static constexpr double SCALE = static_cast<double>(INT64_C(1) << FRAC);

    // Construction
    constexpr Fixed64() noexcept : raw(0) {}
    // Raw-bits constructor (from int64_t): used internally for saturation limits.
    explicit constexpr Fixed64(int64_t r) noexcept : raw(r) {}
    // Value constructor (from int): Fixed64<FRAC>(2) = 2.0 in fixed-point.
    // This makes `S(2)` work in generic Cx<S> / kernel code.
    explicit constexpr Fixed64(int v) noexcept : raw(static_cast<int64_t>(v) << FRAC) {}

    // Explicit conversion to double so static_cast<double>(fx64_val) works in
    // generic kernel code (escape_time.hpp uses scalar_to_double, but having
    // this operator available avoids surprises).
    explicit operator double() const noexcept { return to_double(); }

    // Conversion from double (saturating, round toward zero)
    static Fixed64 from_double(double x) noexcept {
        const double hi = static_cast<double>(std::numeric_limits<int64_t>::max()) / SCALE;
        const double lo = static_cast<double>(std::numeric_limits<int64_t>::min()) / SCALE;
        if (x >= hi) return Fixed64{ std::numeric_limits<int64_t>::max() };
        if (x <= lo) return Fixed64{ std::numeric_limits<int64_t>::min() };
        return Fixed64{ static_cast<int64_t>(x * SCALE) };
    }

    double to_double() const noexcept {
        return static_cast<double>(raw) / SCALE;
    }

    // Arithmetic
    Fixed64 operator+(Fixed64 b) const noexcept { return Fixed64{ raw + b.raw }; }
    Fixed64 operator-(Fixed64 b) const noexcept { return Fixed64{ raw - b.raw }; }
    Fixed64 operator-()        const noexcept {
        return Fixed64{ raw == std::numeric_limits<int64_t>::min()
            ? std::numeric_limits<int64_t>::max()
            : -raw };
    }

    // Multiply: use __int128 to avoid overflow in the product
    Fixed64 operator*(Fixed64 b) const noexcept {
        const __int128 p = static_cast<__int128>(raw) * b.raw;
        const __int128 q = p >> FRAC;
        if (q > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
            return Fixed64{std::numeric_limits<int64_t>::max()};
        }
        if (q < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
            return Fixed64{std::numeric_limits<int64_t>::min()};
        }
        return Fixed64{ static_cast<int64_t>(q) };
    }

    // Square (a == b optimised path)
    Fixed64 sqr() const noexcept {
        const __int128 p = static_cast<__int128>(raw) * raw;
        const __int128 q = p >> FRAC;
        if (q > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
            return Fixed64{std::numeric_limits<int64_t>::max()};
        }
        return Fixed64{ static_cast<int64_t>(q) };
    }

    // Comparisons
    bool operator>=(Fixed64 b) const noexcept { return raw >= b.raw; }
    bool operator<=(Fixed64 b) const noexcept { return raw <= b.raw; }
    bool operator>(Fixed64 b)  const noexcept { return raw >  b.raw; }
    bool operator<(Fixed64 b)  const noexcept { return raw <  b.raw; }
    bool operator==(Fixed64 b) const noexcept { return raw == b.raw; }
};

using FxQ657 = Fixed64<57>;
using FxQ459 = Fixed64<59>;
using FxQ360 = Fixed64<60>;
using Fx64 = FxQ657;

template <typename T>
struct is_fixed64 : std::false_type {};

template <int FRAC>
struct is_fixed64<Fixed64<FRAC>> : std::true_type {};

template <typename T>
inline constexpr bool is_fixed64_v = is_fixed64<T>::value;

// Scalar overload points for Fixed64
template <int FRAC>
inline Fixed64<FRAC> scalar_abs(Fixed64<FRAC> x) noexcept {
    if (x.raw >= 0) return x;
    if (x.raw == std::numeric_limits<int64_t>::min()) {
        return Fixed64<FRAC>{std::numeric_limits<int64_t>::max()};
    }
    return Fixed64<FRAC>{ -x.raw };
}

template <int FRAC>
inline Fixed64<FRAC> scalar_sqrt(Fixed64<FRAC> x) noexcept {
    // Route through fp64 outside the fixed-point render hot loops.
    return Fixed64<FRAC>::from_double(std::sqrt(x.to_double()));
}

// Note: scalar_from_double<Fixed64<FRAC>>(x) is handled by the primary template in
// scalar.hpp which calls S::from_double(x). No separate overload here.
template <int FRAC>
inline double scalar_to_double(Fixed64<FRAC> x) noexcept { return x.to_double(); }

} // namespace fsd::compute
