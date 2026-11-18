#include <stdint.h>
#include <string.h>

#include "geometry.h"
#include "improc_internal.h"
#include "utils.h"

static void flip_h_row(const uchar *restrict s, uchar *restrict d, int width,
                       int cn) {
    const uchar *p = s + (size_t)(width - 1) * (size_t)cn;
    for (int x = 0; x < width; x++, p -= cn, d += cn) {
        for (int c = 0; c < cn; c++) {
            d[c] = p[c];
        }
    }
}

void flip(struct Mat *dst, const struct Mat *src, int flip_code) {
    ASSERT(dst != src && dst->data != src->data);
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
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(x >= 0 && y >= 0);
    // Written as subtractions so x + cols cannot overflow int.
    ASSERT(x <= src->cols && dst->cols <= src->cols - x);
    ASSERT(y <= src->rows && dst->rows <= src->rows - y);
    if (dst->rows == 0 || dst->cols == 0) {
        return;
    }
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

static void resize_linear_row(const uchar *restrict r0,
                              const uchar *restrict r1, uchar *restrict d,
                              int dst_w, int cn, const int *x0, const int *x1,
                              const struct LinearWeights *ax,
                              struct LinearWeights by) {
    for (int x = 0; x < dst_w; x++, d += cn) {
        const uchar *p0 = r0 + (size_t)x0[x] * (size_t)cn;
        const uchar *p1 = r0 + (size_t)x1[x] * (size_t)cn;
        const uchar *q0 = r1 + (size_t)x0[x] * (size_t)cn;
        const uchar *q1 = r1 + (size_t)x1[x] * (size_t)cn;
        int32_t      a0 = ax[x].a0, a1 = ax[x].a1;
        for (int c = 0; c < cn; c++) {
            int32_t top = p0[c] * a0 + p1[c] * a1;
            int32_t bot = q0[c] * a0 + q1[c] * a1;
            int32_t t   = (((top >> 4) * by.a0) >> 16) + (((bot >> 4) * by.a1) >> 16);
            d[c]        = (uchar)((t + 2) >> 2);
        }
    }
}

// ---- float32 emulation for the INTER_LINEAR source positions -------------
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

static void resize_area(struct Mat *dst, const struct Mat *src) {
    int dw = dst->cols, dh = dst->rows, sw = src->cols, sh = src->rows;
    int cn = (int)mat_channel_num(src);
    ASSERT((uint64_t)sw * (uint64_t)sh <= AREA_MAX_SRC_PIXELS);

    // Scratch: two rows of dw*cn uint32_t = 2*dw*cn*4, overflow checked.
    size_t    n    = improc_scratch_mul((size_t)dw, (size_t)cn);
    uint32_t *bufs = fast_malloc(
        improc_scratch_mul(improc_scratch_mul(2, n), sizeof(uint32_t)));
    improc_scratch_check((void *const[]){bufs}, 1);
    uint32_t *hrow = bufs;
    uint32_t *acc  = bufs + n;
    uint32_t  den  = (uint32_t)sw * (uint32_t)sh;
    // OpenCV rounds the mean half to even (cvRound of a float), except an
    // exact 2x2 reduction, whose vector path adds 2 and shifts.
    int half_even = !((int32_t)sw == 2 * (int32_t)dw &&
                      (int32_t)sh == 2 * (int32_t)dh);

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
            if (half_even) {
                q += r > den - r || (r == den - r && (q & 1u));
            } else {
                q += r >= den - r;
            }
            d[i] = (uchar)q;
        }
    }
    fast_free(bufs);
}

void resize(struct Mat *dst, const struct Mat *src, int interpolation) {
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR ||
           interpolation == INTER_AREA);
    ASSERT(mat_depth(src) == CV_8U);
    ASSERT(mat_channel_num(src) == 1 || mat_channel_num(src) == 3 ||
           mat_channel_num(src) == 4);
    ASSERT(mat_type(dst) == mat_type(src));
    if (dst->rows == 0 || dst->cols == 0) {
        return;
    }
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

    // Scratch: coordinate tables 2*dw*sizeof(int) + dw*4, overflow checked.
    // Both sizes are computed before the first allocation, so an overflow
    // ASSERT cannot leave a block behind.
    const size_t xs_bytes =
        improc_scratch_mul(improc_scratch_mul(2, (size_t)dw), sizeof(int));
    const size_t w_bytes =
        improc_scratch_mul((size_t)dw, sizeof(struct LinearWeights));
    int                  *xs  = fast_malloc(xs_bytes);
    struct LinearWeights *fxs = fast_malloc(w_bytes);
    improc_scratch_check((void *const[]){xs, fxs}, 2);
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
        for (int x = 0; x < dw; x++) {
            linear_coord(x, sw, dw, area, 1, &x0[x], &fxs[x]);
            x1[x] = x0[x] + 1 < sw ? x0[x] + 1 : sw - 1;
        }
        for (int y = 0; y < dh; y++) {
            int                  y0;
            struct LinearWeights fy;
            linear_coord(y, sh, dh, area, 0, &y0, &fy);
            int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
            y0     = y0 < 0 ? 0 : y0;
            resize_linear_row(mat_row(src, y0), mat_row(src, y1),
                              mat_row(dst, y), dw, cn, x0, x1, fxs, fy);
        }
    }
    fast_free(xs);
    fast_free(fxs);
}
