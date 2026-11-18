#ifndef IMPROC_SIMD_H_
#define IMPROC_SIMD_H_

// Minimal 128-bit vector layer for the core kernels: SSE2 on x86-64, NEON on
// AArch64 and on 32-bit ARM built with NEON. IMPROC_SIMD is 0 elsewhere (and
// with IMPROC_NO_SIMD), and every kernel then runs its scalar loop. A vector
// path must give the same bytes as the scalar one; the tests run both.
//
// Names: u8 = 16 x uint8, u16 = 8 x uint16, i16 = 8 x int16 (same register
// type as u16), u32 = 4 x uint32. Loads and stores are unaligned.

#include <stdint.h>

#if !defined(IMPROC_NO_SIMD) && (defined(__SSE2__) || defined(_M_X64))
#define IMPROC_SIMD      1
#define IMPROC_SIMD_SSE2 1
#include <emmintrin.h>
typedef __m128i v_u8;
typedef __m128i v_u16;

static inline v_u8 v_ld_u8(const uint8_t *p) {
    return _mm_loadu_si128((const __m128i *)(const void *)p);
}
static inline void v_st_u8(uint8_t *p, v_u8 v) {
    _mm_storeu_si128((__m128i *)(void *)p, v);
}
static inline v_u8 v_min_u8(v_u8 a, v_u8 b) { return _mm_min_epu8(a, b); }
static inline v_u8 v_max_u8(v_u8 a, v_u8 b) { return _mm_max_epu8(a, b); }

static inline v_u16 v_ld_u16(const uint16_t *p) {
    return _mm_loadu_si128((const __m128i *)(const void *)p);
}
static inline void v_st_u16(uint16_t *p, v_u16 v) {
    _mm_storeu_si128((__m128i *)(void *)p, v);
}
static inline v_u16 v_dup_u16(uint16_t x) { return _mm_set1_epi16((short)x); }
static inline v_u16 v_add_u16(v_u16 a, v_u16 b) { return _mm_add_epi16(a, b); }
static inline v_u16 v_mul_u16(v_u16 a, v_u16 b) {
    return _mm_mullo_epi16(a, b);
}
#define v_shr_u16(a, n) _mm_srli_epi16((a), (n))
// Widen the low or high 8 bytes of a to 16 bits.
static inline v_u16 v_lo_u8(v_u8 a) {
    return _mm_unpacklo_epi8(a, _mm_setzero_si128());
}
static inline v_u16 v_hi_u8(v_u8 a) {
    return _mm_unpackhi_epi8(a, _mm_setzero_si128());
}
// Narrow two vectors of values <= 255 to 16 bytes (lo first).
static inline v_u8 v_narrow_u16(v_u16 lo, v_u16 hi) {
    return _mm_packus_epi16(lo, hi);
}
// (a * b) >> 16 per lane, signed 16-bit.
static inline v_u16 v_mulhi_i16(v_u16 a, v_u16 b) {
    return _mm_mulhi_epi16(a, b);
}

// 4 x uint32. lo += a[0..3] * c and hi += a[4..7] * c, 32-bit products.
typedef __m128i v_u32;
static inline v_u32 v_dup_u32(uint32_t x) { return _mm_set1_epi32((int)x); }
static inline void  v_mla_u16_u32(v_u32 *lo, v_u32 *hi, v_u16 a, uint16_t c) {
    __m128i vc = _mm_set1_epi16((short)c);
    __m128i pl = _mm_mullo_epi16(a, vc), ph = _mm_mulhi_epu16(a, vc);
    *lo        = _mm_add_epi32(*lo, _mm_unpacklo_epi16(pl, ph));
    *hi        = _mm_add_epi32(*hi, _mm_unpackhi_epi16(pl, ph));
}
#define v_shr_u32(a, n) _mm_srli_epi32((a), (n))
// Narrow two vectors of values <= 32767 to 8 x uint16 (lo first).
static inline v_u16 v_narrow_u32(v_u32 lo, v_u32 hi) {
    return _mm_packs_epi32(lo, hi);
}

#elif !defined(IMPROC_NO_SIMD) && defined(__ARM_NEON)
#define IMPROC_SIMD      1
#define IMPROC_SIMD_NEON 1
#include <arm_neon.h>
typedef uint8x16_t v_u8;
typedef uint16x8_t v_u16;

static inline v_u8  v_ld_u8(const uint8_t *p) { return vld1q_u8(p); }
static inline void  v_st_u8(uint8_t *p, v_u8 v) { vst1q_u8(p, v); }
static inline v_u8  v_min_u8(v_u8 a, v_u8 b) { return vminq_u8(a, b); }
static inline v_u8  v_max_u8(v_u8 a, v_u8 b) { return vmaxq_u8(a, b); }
static inline v_u16 v_ld_u16(const uint16_t *p) { return vld1q_u16(p); }
static inline void  v_st_u16(uint16_t *p, v_u16 v) { vst1q_u16(p, v); }
static inline v_u16 v_dup_u16(uint16_t x) { return vdupq_n_u16(x); }
static inline v_u16 v_add_u16(v_u16 a, v_u16 b) { return vaddq_u16(a, b); }
static inline v_u16 v_mul_u16(v_u16 a, v_u16 b) { return vmulq_u16(a, b); }
#define v_shr_u16(a, n) vshrq_n_u16((a), (n))
static inline v_u16 v_lo_u8(v_u8 a) { return vmovl_u8(vget_low_u8(a)); }
static inline v_u16 v_hi_u8(v_u8 a) { return vmovl_u8(vget_high_u8(a)); }
static inline v_u8  v_narrow_u16(v_u16 lo, v_u16 hi) {
    return vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
}
static inline v_u16 v_mulhi_i16(v_u16 a, v_u16 b) {
    int16x8_t sa = vreinterpretq_s16_u16(a), sb = vreinterpretq_s16_u16(b);
    int32x4_t lo = vmull_s16(vget_low_s16(sa), vget_low_s16(sb));
    int32x4_t hi = vmull_s16(vget_high_s16(sa), vget_high_s16(sb));
    return vreinterpretq_u16_s16(
        vcombine_s16(vshrn_n_s32(lo, 16), vshrn_n_s32(hi, 16)));
}

typedef uint32x4_t v_u32;
static inline v_u32 v_dup_u32(uint32_t x) { return vdupq_n_u32(x); }
static inline void  v_mla_u16_u32(v_u32 *lo, v_u32 *hi, v_u16 a, uint16_t c) {
    *lo = vmlal_n_u16(*lo, vget_low_u16(a), c);
    *hi = vmlal_n_u16(*hi, vget_high_u16(a), c);
}
#define v_shr_u32(a, n) vshrq_n_u32((a), (n))
static inline v_u16 v_narrow_u32(v_u32 lo, v_u32 hi) {
    return vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
}

#else
#define IMPROC_SIMD 0
#endif

#endif // IMPROC_SIMD_H_
