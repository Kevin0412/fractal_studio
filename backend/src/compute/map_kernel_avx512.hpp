// compute/map_kernel_avx512.hpp
//
// AVX-512 accelerated map renderer.
//
// Processes 8 pixels simultaneously using AVX-512 double-precision (zmm
// registers, __m512d). Escape check is done in bulk with masked iteration —
// pixels that have already escaped have their lanes masked off, keeping the
// remaining active lanes churning. This is especially effective when the
// iteration counts across a tile are relatively uniform.
//
// Fx64 on AVX-512: the 512-bit integer path uses __m512i (8 × int64_t) with
// IFMA52 instructions (_mm512_madd52lo/hi_epu64) for the 52-bit partial
// products of the 57-bit fixed-point multiplication. The upper 5 bits are
// covered by a separate __int128 scalar correction — this is still much faster
// than 8 sequential __int128 muls.
//
// Feature detection: callers must check avx512_available() before calling.
// If AVX-512 is not available at runtime the map_kernel.cpp OpenMP path is
// used instead.

#pragma once

#include "map_kernel.hpp"

namespace fsd::compute {

// Returns true if the CPU supports AVX-512F at runtime.
bool avx512_available() noexcept;

// AVX-512 fp64 render (8 pixels at a time).
// Pre-condition: avx512_available() == true.
MapStats render_map_avx512_fp64(const MapParams& p, cv::Mat& out);

// AVX-512 Fx64 render (8 pixels at a time with IFMA52).
// Pre-condition: avx512_available() == true.
MapStats render_map_avx512_fx64(const MapParams& p, cv::Mat& out);

} // namespace fsd::compute
