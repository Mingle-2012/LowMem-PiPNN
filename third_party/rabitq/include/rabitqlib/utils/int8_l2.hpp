#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

namespace rabitqlib {
// RAW-QG retains its float ABI. Signed byte values are losslessly carried in
// those slots; widen to int32 BEFORE subtraction and accumulate integer squares.
// No extra N*D byte copy or per-thread N-sized state is needed.
static inline bool signed_int8_input() {
#ifdef LOWMEM_INT8_L2
    static const bool enabled = [] { const char* p = std::getenv("LOWMEM_INPUT_DTYPE");
        return p && std::strcmp(p, "int8") == 0; }();
    return enabled;
#else
    return false;
#endif
}
inline float int8_l2_carrier(const float* a, const float* b, size_t d) {
    size_t i = 0;
    int64_t total = 0;
#ifdef __AVX512F__
    __m512i sums = _mm512_setzero_si512();
    for (; i + 16 <= d; i += 16) {
        auto delta = _mm512_sub_epi32(_mm512_cvttps_epi32(_mm512_loadu_ps(a+i)),
                                     _mm512_cvttps_epi32(_mm512_loadu_ps(b+i)));
        sums = _mm512_add_epi32(sums, _mm512_mullo_epi32(delta, delta));
    }
    alignas(64) int32_t lanes[16]; _mm512_store_si512(lanes, sums);
    for (auto value : lanes) total += value;
#elif defined(__AVX2__)
    __m256i sums = _mm256_setzero_si256();
    for (; i + 8 <= d; i += 8) {
        auto delta = _mm256_sub_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(a+i)),
                                     _mm256_cvttps_epi32(_mm256_loadu_ps(b+i)));
        sums = _mm256_add_epi32(sums, _mm256_mullo_epi32(delta, delta));
    }
    alignas(32) int32_t lanes[8]; _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sums);
    for (auto value : lanes) total += value;
#endif
    for (; i < d; ++i) { int32_t delta = int32_t(a[i])-int32_t(b[i]); total += delta*delta; }
    return float(total);
}
} // namespace rabitqlib
