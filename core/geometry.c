#include <stdint.h>
#include <string.h>

#include "geometry.h"
#include "improc_internal.h"
#include "improc_simd.h"
#include "utils.h"

// An offset, not a pointer, walks back: s - cn after the last pixel would be
// undefined. Inlined with a constant cn the inner loop becomes plain moves.
static inline void flip_h_cn(const uchar *restrict s, uchar *restrict d,
                             int width, int cn) {
    size_t off = (size_t)width * (size_t)cn;
    for (int x = 0; x < width; x++, d += cn) {
        off -= (size_t)cn;
        for (int c = 0; c < cn; c++) {
            d[c] = s[off + (size_t)c];
        }
    }
}

static void flip_h_row(const uchar *restrict s, uchar *restrict d, int width,
                       int cn) {
    switch (cn) {
    case 1:
        flip_h_cn(s, d, width, 1);
        break;
    case 3:
        flip_h_cn(s, d, width, 3);
        break;
    case 4:
        flip_h_cn(s, d, width, 4);
        break;
    default:
        flip_h_cn(s, d, width, cn);
        break;
    }
}

void flip(struct Mat *dst, const struct Mat *src, int flip_code) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(flip_code == FLIP_VERTICAL || flip_code == FLIP_HORIZONTAL ||
           flip_code == FLIP_BOTH);
    ASSERT(mat_depth(src) == CV_8U);
    ASSERT(mat_channel_num(src) == 1 || mat_channel_num(src) == 3 ||
           mat_channel_num(src) == 4);
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(src->rows == dst->rows && src->cols == dst->cols);
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    int    cn         = (int)mat_channel_num(src);
    size_t row_bytes  = (size_t)src->cols * (size_t)cn;
    int    horizontal = flip_code != FLIP_VERTICAL;
    int    vertical   = flip_code != FLIP_HORIZONTAL;
    for (int y = 0; y < src->rows; y++) {
        const uchar *s = mat_row(src, vertical ? src->rows - 1 - y : y);
        uchar       *d = mat_row(dst, y);
        if (horizontal) {
            flip_h_row(s, d, src->cols, cn);
        } else {
            memcpy(d, s, row_bytes);
        }
    }
}

void crop(struct Mat *dst, const struct Mat *src, int x, int y) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(x >= 0 && y >= 0);
    // Written as subtractions so x + cols cannot overflow int.
    ASSERT(x <= src->cols && dst->cols <= src->cols - x);
    ASSERT(y <= src->rows && dst->rows <= src->rows - y);
    if (dst->rows == 0 || dst->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    size_t px        = mat_pixel_size(src);
    size_t row_bytes = (size_t)dst->cols * px;
    size_t offset    = (size_t)x * px;
    for (int r = 0; r < dst->rows; r++) {
        memcpy(mat_row(dst, r), mat_row(src, y + r) + offset, row_bytes);
    }
}

static void resize_nearest_row(const uchar *restrict s, uchar *restrict d,
                               int dst_w, int cn, const int *sx) {
    for (int x = 0; x < dst_w; x++, d += cn) {
        const uchar *p = s + (size_t)sx[x] * (size_t)cn;
        for (int c = 0; c < cn; c++) {
            d[c] = p[c];
        }
    }
}

// OpenCV's INTER_LINEAR for 8-bit images. Weights are Q11 and sum to 2048 up
// to one unit of rounding (OpenCV rounds the two weights separately). The
// horizontal sum h = a0 * p0 + a1 * p1 is at most 255 * 2049. The vertical
// step follows OpenCV's vector code: each row term is ((h >> 4) * b) >> 16,
// and the two terms are added and rounded with (t + 2) >> 2. (h >> 4) * b is
// below 2^31, and nothing negative is shifted.
#define LINEAR_ONE 2048

struct LinearWeights {
    uint16_t a0; // weight of pixel i
    uint16_t a1; // weight of pixel i + 1
};

// Horizontal pass of one source row: h[x * cn + c] = (a0 p0 + a1 p1) >> 4,
// the only form the vertical step reads. At most 255 * 2049 >> 4 = 32655, so
// int16_t.
static void linear_hrow(const uchar *restrict s, int16_t *restrict h,
                        int dst_w, int cn, const int *x0, const int *x1,
                        const struct LinearWeights *ax) {
    for (int x = 0; x < dst_w; x++, h += cn) {
        const uchar *p0 = s + (size_t)x0[x] * (size_t)cn;
        const uchar *p1 = s + (size_t)x1[x] * (size_t)cn;
        int32_t      a0 = ax[x].a0, a1 = ax[x].a1;
        for (int c = 0; c < cn; c++) {
            h[c] = (int16_t)((p0[c] * a0 + p1[c] * a1) >> 4);
        }
    }
}

// Vertical step: ((h0 * b0) >> 16) + ((h1 * b1) >> 16), then (t + 2) >> 2.
// Every term is non-negative and t <= 1021.
static void linear_vrow(const int16_t *restrict h0, const int16_t *restrict h1,
                        uchar *restrict d, size_t n, struct LinearWeights by) {
    const int32_t b0 = by.a0, b1 = by.a1;
    size_t        i  = 0;
#if IMPROC_SIMD
    const v_u16 vb0 = v_dup_u16((uint16_t)b0), vb1 = v_dup_u16((uint16_t)b1);
    const v_u16 two = v_dup_u16(2);
    for (; i + 16 <= n; i += 16) {
        const uint16_t *u0 = (const uint16_t *)h0 + i;
        const uint16_t *u1 = (const uint16_t *)h1 + i;
        v_u16 lo = v_add_u16(v_mulhi_i16(v_ld_u16(u0), vb0),
                             v_mulhi_i16(v_ld_u16(u1), vb1));
        v_u16 hi = v_add_u16(v_mulhi_i16(v_ld_u16(u0 + 8), vb0),
                             v_mulhi_i16(v_ld_u16(u1 + 8), vb1));
        lo       = v_shr_u16(v_add_u16(lo, two), 2);
        hi       = v_shr_u16(v_add_u16(hi, two), 2);
        v_st_u8(d + i, v_narrow_u16(lo, hi));
    }
#endif
    for (; i < n; i++) {
        int32_t t = ((h0[i] * b0) >> 16) + ((h1[i] * b1) >> 16);
        d[i]      = (uchar)((t + 2) >> 2);
    }
}

// OpenCV computes fx = (float)((dx + 0.5) * scale - 0.5) in double and float,
// then sx = floor(fx), fx -= sx, a1 = cvRound(fx * 2048) and
// a0 = cvRound((1.f - fx) * 2048). Here fx is the float32 nearest to the
// exact position ((2 dx + 1) sw - dw) / (2 dw), ties to even, and every
// float32 value is the exact fraction num / 2^sh. That equals OpenCV except
// when the exact position is itself a float32 midpoint: OpenCV's double
// error then picks the side. This never happens up to 4096 pixels and stays
// below 0.1% of the samples in the larger sizes measured.

// Rounds u / d (d > 0) to 24 significant bits, ties to even, as m / 2^sh.
// Only called with d = 2 dw <= 2^32. When the numerator is shifted up it ends below 2^(25 + 33),
// and when the denominator is, it ends below 2^(64 - 23), so nothing wraps.
static void f32_ratio(uint64_t u, uint64_t d, uint64_t *m, int *sh) {
    if (u == 0) {
        *m  = 0;
        *sh = 0;
        return;
    }
    int bu = 0, bd = 0;
    while ((u >> bu) != 0) {
        bu++;
    }
    while ((d >> bd) != 0) {
        bd++;
    }
    // u / d lies in [2^(bu - bd - 1), 2^(bu - bd + 1)); aim for 24 bits.
    int s = 24 - (bu - bd);
    for (;;) {
        uint64_t num = s >= 0 ? u << s : u;
        uint64_t den = s >= 0 ? d : d << -s;
        uint64_t q   = num / den;
        if (q >= ((uint64_t)1 << 24)) {
            s--;
            continue;
        }
        if (q < ((uint64_t)1 << 23)) {
            s++;
            continue;
        }
        uint64_t r = num - q * den;
        q += r > den - r || (r == den - r && (q & 1u));
        *m  = q; // may be 2^24, still exact in float32
        *sh = s;
        return;
    }
}

// Rounds u / 2^sh (u < 2^63) to 24 significant bits, ties to even, as
// m / 2^s. Shifts only, so any sh is safe; a u of 24 bits or fewer is exact
// and is returned unchanged.
static void f32_pow2(uint64_t u, int sh, uint64_t *m, int *s) {
    int n = 0;
    while ((u >> n) >= ((uint64_t)1 << 24)) {
        n++;
    }
    if (n > 0) {
        uint64_t q = u >> n, r = u & (((uint64_t)1 << n) - 1u);
        uint64_t half = (uint64_t)1 << (n - 1);
        q += r > half || (r == half && (q & 1u));
        u = q;
    }
    *m = u;
    *s = sh - n;
}

// round(m / 2^sh * 2048), ties to even, for 0 <= m / 2^sh <= 1.
static uint16_t q11_round(uint64_t m, int sh) {
    int k = sh - 11; // m * 2048 / 2^sh = m / 2^k
    if (k <= 0) {
        return (uint16_t)(m << -k);
    }
    if (k >= 63) {
        return 0;
    }
    uint64_t q = m >> k, r = m & (((uint64_t)1 << k) - 1u);
    uint64_t half = (uint64_t)1 << (k - 1);
    q += r > half || (r == half && (q & 1u));
    return (uint16_t)q;
}

// Weights from the fractional part f = F / 2^sh, 0 <= f < 1, which is
// already a float32 value: a1 = round(f * 2048), a0 = round(float32(1 - f) *
// 2048).
static struct LinearWeights q11_weights(uint64_t F, int sh) {
    struct LinearWeights w;
    if (F == 0) {
        w.a0 = LINEAR_ONE;
        w.a1 = 0;
        return w;
    }
    w.a1 = q11_round(F, sh);
    // A nonzero position is at least 1 / (2 dw) >= 2^-32, so sh <= 56 and
    // 2^sh fits 64 bits.
    uint64_t one = (uint64_t)1 << sh;
    uint64_t m;
    int      s;
    f32_pow2(one - F, sh, &m, &s);
    w.a0 = q11_round(m, s);
    return w;
}

// Source index i and the weights of pixels i and i + 1 for destination index
// dx. INTER_LINEAR follows OpenCV's float32 position (above). INTER_AREA when
// upscaling uses OpenCV's area mapping: i = floor(dx sw / dw), and pixel i + 1
// gets the part of destination cell dx past source cell i,
// max(0, (dx + 1) sw - (i + 1) dw) / sw, rounded half to even in Q11 (OpenCV
// rounds that fraction to float32 first, which moves a weight by one unit
// only at sizes around 10000 and more). With
// clamp (the x axis), a position left of pixel 0 or at or past pixel sw - 1
// takes that edge pixel with weight 0. Without clamp (the y axis, as in
// OpenCV), i may be -1 or sw - 1 and the caller clamps the two row indices.
static void linear_coord(int dx, int sw, int dw, int area, int clamp, int *x0,
                         struct LinearWeights *w) {
    int64_t i;
    if (area) {
        i           = (int64_t)dx * sw / dw;
        int64_t num = ((int64_t)dx + 1) * sw - (i + 1) * dw;
        int64_t f   = 0;
        if (num > 0) {
            int64_t r = num * LINEAR_ONE;
            f         = r / sw;
            int64_t m = r - f * sw;
            f += 2 * m > sw || (2 * m == sw && (f & 1));
        }
        w->a1 = (uint16_t)f;
        w->a0 = (uint16_t)(LINEAR_ONE - f);
    } else {
        int64_t  n = (2 * (int64_t)dx + 1) * sw - dw;
        uint64_t m;
        int      sh;
        f32_ratio((uint64_t)(n < 0 ? -n : n), 2 * (uint64_t)dw, &m, &sh);
        if (sh <= 0) { // |fx| >= 2^23: a whole number
            i  = (int64_t)(m << -sh) * (n < 0 ? -1 : 1);
            *w = q11_weights(0, 0);
        } else if (n >= 0) {
            i  = (int64_t)(m >> sh);
            *w = q11_weights(m - ((uint64_t)i << sh), sh);
        } else {
            // -1/2 < fx < 0 here, since -dw < n < 0: floor is -1, and the
            // fraction fx + 1 is rounded to float32 again. When |fx| <=
            // 2^-25 (only for dw > 2^24) it rounds to 1.0, which gives
            // i = -1 with weights (0, 2048), as in OpenCV.
            uint64_t fm;
            int      fs;
            f32_pow2(((uint64_t)1 << sh) - m, sh, &fm, &fs);
            i  = -1;
            *w = q11_weights(fm, fs);
        }
    }
    // Without clamp the caller clamps both row indices to sw - 1, so an index
    // past it reads the same rows; above 2^24 rows float32 positions can land
    // there (as in OpenCV), and the index must also fit an int.
    if (!clamp && i > sw - 1) {
        i = sw - 1;
    }
    if (clamp && (i < 0 || i >= sw - 1)) {
        i  = i < 0 ? 0 : sw - 1;
        *w = q11_weights(0, 0);
    }
    *x0 = (int)i;
}

void improc_resize_coord(int dx, int sw, int dw, int area, int clamp, int *x0,
                         uint16_t *a0, uint16_t *a1) {
    struct LinearWeights w;
    linear_coord(dx, sw, dw, area, clamp, x0, &w);
    *a0 = w.a0;
    *a1 = w.a1;
}

// Largest source pixel count INTER_AREA accepts: 255 * 2^24 < 2^32.
#define AREA_MAX_SRC_PIXELS ((uint32_t)1 << 24)

// Horizontal pass of INTER_AREA for one source row: out[x*cn+c] is the sum of
// weight*pixel over the source pixels under destination pixel x. A weight is
// at most src_w <= 2^24, so a term is at most 255*2^24 and a row sum at most
// 255*src_w, both below 2^32.
static void area_row_h(const uchar *restrict s, uint32_t *restrict out,
                       int dst_w, int src_w, int cn) {
    uint64_t sw = (uint64_t)src_w;
    uint64_t dw = (uint64_t)dst_w;
    uint64_t k  = 0;
    for (int x = 0; x < dst_w; x++, out += cn) {
        uint64_t pos = (uint64_t)x * sw;
        uint64_t end = pos + sw;
        for (int c = 0; c < cn; c++) {
            out[c] = 0;
        }
        while (pos < end) {
            uint64_t     k_end = (k + 1) * dw;
            uint64_t     stop  = k_end < end ? k_end : end;
            uint32_t     w     = (uint32_t)(stop - pos);
            const uchar *p     = s + (size_t)k * (size_t)cn;
            for (int c = 0; c < cn; c++) {
                out[c] += w * (uint32_t)p[c];
            }
            pos = stop;
            if (pos == k_end) {
                k++;
            }
        }
    }
}

// Exact 2x2 reduction: each output is (a + b + c + d + 2) >> 2, the half-up
// mean OpenCV's vector path gives for this case. No scratch.
static void resize_area_half(struct Mat *dst, const struct Mat *src) {
    const size_t ucn = mat_channel_num(src);
    for (int y = 0; y < dst->rows; y++) {
        const uchar *s0 = mat_row(src, 2 * y);
        const uchar *s1 = mat_row(src, 2 * y + 1);
        uchar       *d  = mat_row(dst, y);
        for (int x = 0; x < dst->cols;
             x++, s0 += 2 * ucn, s1 += 2 * ucn, d += ucn) {
            for (size_t c = 0; c < ucn; c++) {
                unsigned v = (unsigned)s0[c] + s0[c + ucn] + s1[c] + s1[c + ucn];
                d[c]       = (uchar)((v + 2u) >> 2);
            }
        }
    }
}

static void resize_area(struct Mat *dst, const struct Mat *src) {
    int dw = dst->cols, dh = dst->rows, sw = src->cols, sh = src->rows;
    int cn = (int)mat_channel_num(src);
    ASSERT((uint64_t)sw * (uint64_t)sh <= AREA_MAX_SRC_PIXELS);

    if ((int32_t)sw == 2 * (int32_t)dw && (int32_t)sh == 2 * (int32_t)dh) {
        resize_area_half(dst, src);
        return;
    }

    size_t    n    = improc_scratch_mul((size_t)dw, (size_t)cn);
    uint32_t *bufs = fast_malloc(
        improc_scratch_mul(improc_scratch_mul(2, n), sizeof(uint32_t)));
    improc_scratch_check((void *const[]){bufs}, 1);
    uint32_t *hrow = bufs;
    uint32_t *acc  = bufs + n;
    uint32_t  den  = (uint32_t)sw * (uint32_t)sh;
    // OpenCV rounds the mean half to even (cvRound of a float); the exact 2x2
    // reduction, which rounds half up, was handled above.

    uint64_t src_h  = (uint64_t)sh;
    uint64_t dst_h  = (uint64_t)dh;
    uint64_t k      = 0;
    int      cached = -1; // source row currently held in hrow
    for (int y = 0; y < dh; y++) {
        uint64_t pos = (uint64_t)y * src_h;
        uint64_t end = pos + src_h;
        for (size_t i = 0; i < n; i++) {
            acc[i] = 0;
        }
        while (pos < end) {
            uint64_t k_end = (k + 1) * dst_h;
            uint64_t stop  = k_end < end ? k_end : end;
            uint32_t w     = (uint32_t)(stop - pos);
            if (cached != (int)k) {
                area_row_h(mat_row(src, (int)k), hrow, dw, sw, cn);
                cached = (int)k;
            }
            for (size_t i = 0; i < n; i++) {
                acc[i] += w * hrow[i];
            }
            pos = stop;
            if (pos == k_end) {
                k++;
            }
        }
        uchar *d = mat_row(dst, y);
        for (size_t i = 0; i < n; i++) {
            uint32_t q = acc[i] / den;
            uint32_t r = acc[i] - q * den;
            q += r > den - r || (r == den - r && (q & 1u));
            d[i] = (uchar)q;
        }
    }
    fast_free(bufs);
}

// Linear kernel (INTER_LINEAR, or INTER_AREA when an axis grows). Each source
// row's horizontal pass is computed once and kept in one of two buffers, so
// consecutive output rows that read the same source rows reuse it. hbuf holds
// two rows of dst->cols * channels values.
static void resize_linear(struct Mat *dst, const struct Mat *src, int area,
                          int *x0, int *x1, struct LinearWeights *ax,
                          int16_t *hbuf) {
    int      dw = dst->cols, dh = dst->rows, sw = src->cols, sh = src->rows;
    int      cn = (int)mat_channel_num(src);
    size_t   n  = (size_t)dw * (size_t)cn; // checked by the caller's scratch size
    int16_t *slot[2] = {hbuf, hbuf + n};
    int      tag[2]  = {-1, -1}; // source row held by each slot

    for (int x = 0; x < dw; x++) {
        linear_coord(x, sw, dw, area, 1, &x0[x], &ax[x]);
        x1[x] = x0[x] + 1 < sw ? x0[x] + 1 : sw - 1;
    }
    for (int y = 0; y < dh; y++) {
        int                  y0;
        struct LinearWeights fy;
        linear_coord(y, sh, dh, area, 0, &y0, &fy);
        int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
        y0     = y0 < 0 ? 0 : y0;
        const int16_t *h[2];
        for (int k = 0; k < 2; k++) {
            int row = k == 0 ? y0 : y1;
            int s   = tag[0] == row ? 0 : tag[1] == row ? 1 : -1;
            if (s < 0) {
                // Refill a slot that does not hold the other row needed now.
                s = (k == 1 && tag[0] == y0) ? 1 : 0;
                linear_hrow(mat_row(src, row), slot[s], dw, cn, x0, x1, ax);
                tag[s] = row;
            }
            h[k] = slot[s];
        }
        linear_vrow(h[0], h[1], mat_row(dst, y), n, fy);
    }
}

void resize(struct Mat *dst, const struct Mat *src, int interpolation) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR ||
           interpolation == INTER_AREA);
    ASSERT(mat_depth(src) == CV_8U);
    ASSERT(mat_channel_num(src) == 1 || mat_channel_num(src) == 3 ||
           mat_channel_num(src) == 4);
    ASSERT(mat_type(dst) == mat_type(src));
    if (dst->rows == 0 || dst->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    ASSERT(src->rows > 0 && src->cols > 0);

    // OpenCV averages areas only when neither axis grows; otherwise
    // INTER_AREA runs the linear kernel with the area mapping.
    int area = interpolation == INTER_AREA;
    if (area && dst->cols <= src->cols && dst->rows <= src->rows) {
        resize_area(dst, src);
        return;
    }

    int dw = dst->cols, dh = dst->rows, sw = src->cols, sh = src->rows;
    int cn = (int)mat_channel_num(src);

    // Every size is computed before the first allocation, so an overflow
    // ASSERT cannot leave a block behind.
    const size_t xs_bytes =
        improc_scratch_mul(improc_scratch_mul(2, (size_t)dw), sizeof(int));
    const size_t w_bytes =
        improc_scratch_mul((size_t)dw, sizeof(struct LinearWeights));
    const size_t h_bytes =
        interpolation == INTER_NEAREST
            ? 0
            : improc_scratch_mul(
                  improc_scratch_mul(improc_scratch_mul(2, (size_t)dw),
                                     (size_t)cn),
                  sizeof(int16_t));
    int                  *xs   = fast_malloc(xs_bytes);
    struct LinearWeights *fxs  = fast_malloc(w_bytes);
    int16_t              *hbuf = NULL;
    if (interpolation == INTER_NEAREST) {
        improc_scratch_check((void *const[]){xs, fxs}, 2);
    } else {
        hbuf = fast_malloc(h_bytes);
        improc_scratch_check((void *const[]){xs, fxs, hbuf}, 3);
    }
    int *x0 = xs, *x1 = xs + dw;

    if (interpolation == INTER_NEAREST) {
        for (int x = 0; x < dw; x++) {
            x0[x] =
                (int)(((int64_t)(2 * (int64_t)x + 1) * sw) / (2 * (int64_t)dw));
        }
        for (int y = 0; y < dh; y++) {
            int sy =
                (int)(((int64_t)(2 * (int64_t)y + 1) * sh) / (2 * (int64_t)dh));
            resize_nearest_row(mat_row(src, sy), mat_row(dst, y), dw, cn, x0);
        }
    } else {
        resize_linear(dst, src, area, x0, x1, fxs, hbuf);
    }
    fast_free(hbuf);
    fast_free(xs);
    fast_free(fxs);
}
