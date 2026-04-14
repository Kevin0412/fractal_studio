// compute/complex.hpp
//
// Minimal complex number type templated on scalar. We do not use std::complex
// because:
//   1. In Phase 3 we want to instantiate with Fx64, which has no operator
//      overloads for the std::complex template contract.
//   2. Kernels need very specific sequences of real/imag operations (conj,
//      abs-of-component, abs-of-product) that are clearer on a small hand-
//      rolled type.
//
// This type is POD-ish and all methods are constexpr where possible so the
// compiler inlines everything into the iteration loops.

#pragma once

#include "scalar.hpp"

namespace fsd::compute {

template <typename S>
struct Cx {
    S re;
    S im;

    constexpr Cx() : re(S{}), im(S{}) {}
    constexpr Cx(S r, S i) : re(r), im(i) {}

    constexpr Cx operator+(const Cx& o) const { return {re + o.re, im + o.im}; }
    constexpr Cx operator-(const Cx& o) const { return {re - o.re, im - o.im}; }

    // Full complex multiply: (a+bi)(c+di) = (ac - bd) + (ad + bc)i
    constexpr Cx operator*(const Cx& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }

    // Squaring, cheaper than general multiply: (a+bi)² = (a²-b²) + 2ab i
    constexpr Cx sqr() const {
        return {re * re - im * im, S(2) * re * im};
    }

    constexpr S norm2() const { return re * re + im * im; }
};

template <typename S>
inline S cabs(const Cx<S>& z) {
    return scalar_sqrt(z.norm2());
}

// Conjugate-squared: conj(z²) = (a²-b²) - 2ab i (tricorn variant).
template <typename S>
constexpr Cx<S> conj_sqr(const Cx<S>& z) {
    return {z.re * z.re - z.im * z.im, -(S(2) * z.re * z.im)};
}

} // namespace fsd::compute
