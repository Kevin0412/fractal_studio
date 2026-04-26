// compute/cpu_features.cpp

#include "cpu_features.hpp"

#if defined(__x86_64__) || defined(__i386__)
#  include <cpuid.h>
#endif

#include <cstdint>

namespace fsd::compute {
namespace {

#if defined(__x86_64__) || defined(__i386__)
uint64_t xgetbv0() noexcept {
    unsigned int eax = 0, edx = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
#endif
    return (static_cast<uint64_t>(edx) << 32) | eax;
}

bool cpuid_leaf(unsigned int leaf, unsigned int subleaf,
                unsigned int& eax, unsigned int& ebx,
                unsigned int& ecx, unsigned int& edx) noexcept {
    return __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx) != 0;
}

bool os_avx_state_available() noexcept {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!cpuid_leaf(1, 0, eax, ebx, ecx, edx)) return false;
    const bool has_xsave = static_cast<bool>((ecx >> 26) & 1u);
    const bool has_osxsave = static_cast<bool>((ecx >> 27) & 1u);
    const bool has_avx = static_cast<bool>((ecx >> 28) & 1u);
    if (!has_xsave || !has_osxsave || !has_avx) return false;
    const uint64_t xcr0 = xgetbv0();
    return (xcr0 & 0x6u) == 0x6u; // XMM + YMM state.
}
#endif

} // namespace

bool avx2_available() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    if (!os_avx_state_available()) return false;
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!cpuid_leaf(7, 0, eax, ebx, ecx, edx)) return false;
    return static_cast<bool>((ebx >> 5) & 1u);
#else
    return false;
#endif
}

bool fma_available() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    if (!os_avx_state_available()) return false;
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!cpuid_leaf(1, 0, eax, ebx, ecx, edx)) return false;
    return static_cast<bool>((ecx >> 12) & 1u);
#else
    return false;
#endif
}

bool bmi2_available() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!cpuid_leaf(7, 0, eax, ebx, ecx, edx)) return false;
    return static_cast<bool>((ebx >> 8) & 1u);
#else
    return false;
#endif
}

bool avx512_os_state_available() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    if (!os_avx_state_available()) return false;
    const uint64_t xcr0 = xgetbv0();
    return (xcr0 & 0xE6u) == 0xE6u; // XMM, YMM, opmask, ZMM_hi256, hi16_ZMM.
#else
    return false;
#endif
}

} // namespace fsd::compute
