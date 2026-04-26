// compute/cpu_features.hpp
//
// Runtime CPU feature probes used by SIMD dispatch. This file is intentionally
// compiled without ISA-specific flags so feature checks are safe on older CPUs.

#pragma once

namespace fsd::compute {

bool avx2_available() noexcept;
bool fma_available() noexcept;
bool bmi2_available() noexcept;
bool avx512_os_state_available() noexcept;

} // namespace fsd::compute
