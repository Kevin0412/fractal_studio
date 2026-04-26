// compute/map_kernel_avx2.hpp
//
// AVX2/FMA fp64 map renderer. Processes 4 pixels at a time with __m256d.

#pragma once

#include "cpu_features.hpp"
#include "map_kernel.hpp"

namespace fsd::compute {

MapStats render_map_avx2_fp64(const MapParams& p, cv::Mat& out);

} // namespace fsd::compute
