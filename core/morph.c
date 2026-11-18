#include "morph.h"

#include <stdint.h>

#include "improc_internal.h"
#include "improc_simd.h"
#include "utils.h"

static int is_u8_image_type(const struct Mat *m) {
    size_t t = mat_type(m);
    return t == (size_t)CV_8UC1 || t == (size_t)CV_8UC3 || t == (size_t)CV_8UC4;
}

// acc[i] = min/max(acc[i], row[i]) for n bytes.
static void vminmax_row(uchar *restrict acc, const uchar *restrict row, int n,
                        int is_max) {
    int i = 0;
#if IMPROC_SIMD
    for (; i + 16 <= n; i += 16) {
        v_u8 a = v_ld_u8(acc + i), b = v_ld_u8(row + i);
        v_st_u8(acc + i, is_max ? v_max_u8(a, b) : v_min_u8(a, b));
    }
#endif
    for (; i < n; i++) {
        uchar a = acc[i], b = row[i];
        acc[i]  = is_max ? (b > a ? b : a) : (b < a ? b : a);
    }
}

#if IMPROC_SIMD
// Horizontal pass for the columns whose window lies inside the row, 16 bytes
// at a time: out[j] = min/max of in[j + t * cn], t = -r..r. Returns the first
// column left for the scalar code (r when the interior is narrower than one
// vector, so the scalar code then does every column).
static int hminmax_simd(const uchar *restrict in, uchar *restrict out,
                        int cols, int cn, int r, int is_max) {
    if (cols <= 2 * r) {
        return r;
    }
    const size_t ucn = (size_t)cn;
    const size_t lo  = (size_t)r * ucn;
    const size_t hi  = (size_t)(cols - r) * ucn;
    if (hi - lo < 16) {
        return r;
    }
    for (size_t j = lo;; j += 16) {
        if (j + 16 > hi) {
            j = hi - 16; // last block overlaps the previous one
        }
        const uchar *p = in + j - lo;
        v_u8         m = v_ld_u8(p);
        for (int t = 1; t <= 2 * r; t++) {
            v_u8 v = v_ld_u8(p + (size_t)t * ucn);
            m      = is_max ? v_max_u8(m, v) : v_min_u8(m, v);
        }
        v_st_u8(out + j, m);
        if (j + 16 >= hi) {
            break;
        }
    }
    return cols - r;
}
#endif

// Horizontal pass: out[x] = min/max of in[x - r .. x + r], clipped to
// [0, cols - 1], per channel, for x0 <= x < x1. in and out hold cols * cn
// bytes and are distinct.
static void hmin_cols(const uchar *restrict in, uchar *restrict out, int cols,
                      int cn, int r, int x0, int x1) {
    for (int x = x0; x < x1; x++) {
        const int lo = x - r < 0 ? 0 : x - r;
        const int hi = (int32_t)x + r > cols - 1 ? cols - 1 : x + r;
        for (int c = 0; c < cn; c++) {
            uchar m = in[(size_t)lo * (size_t)cn + (size_t)c];
            for (int i = lo + 1; i <= hi; i++) {
                uchar v = in[(size_t)i * (size_t)cn + (size_t)c];
                m       = v < m ? v : m;
            }
            out[(size_t)x * (size_t)cn + (size_t)c] = m;
        }
    }
}

static void hmax_cols(const uchar *restrict in, uchar *restrict out, int cols,
                      int cn, int r, int x0, int x1) {
    for (int x = x0; x < x1; x++) {
        const int lo = x - r < 0 ? 0 : x - r;
        const int hi = (int32_t)x + r > cols - 1 ? cols - 1 : x + r;
        for (int c = 0; c < cn; c++) {
            uchar m = in[(size_t)lo * (size_t)cn + (size_t)c];
            for (int i = lo + 1; i <= hi; i++) {
                uchar v = in[(size_t)i * (size_t)cn + (size_t)c];
                m       = v > m ? v : m;
            }
            out[(size_t)x * (size_t)cn + (size_t)c] = m;
        }
    }
}

static void hminmax_row(const uchar *restrict in, uchar *restrict out,
                        int cols, int cn, int r, int is_max) {
    void (*cols_fn)(const uchar *restrict, uchar *restrict, int, int, int, int,
                    int) = is_max ? hmax_cols : hmin_cols;
    int done = r; // columns [r, done) already written
#if IMPROC_SIMD
    done = hminmax_simd(in, out, cols, cn, r, is_max);
#endif
    if (done > r) {
        cols_fn(in, out, cols, cn, r, 0, r);
        cols_fn(in, out, cols, cn, r, done, cols);
    } else {
        cols_fn(in, out, cols, cn, r, 0, cols);
    }
}

static void morph(struct Mat *dst, const struct Mat *src, int ksize,
                  int is_dilate) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(is_u8_image_type(src));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(ksize >= 1 && ksize <= 255 && (ksize & 1));
    if (src->rows == 0 || src->cols == 0) {
        return;
    }

    const int rows = src->rows;
    const int cn   = (int)mat_channel_num(src);
    const int n    = src->cols * cn;
    const int r    = ksize / 2;

    // Scratch: one row of n = cols*cn bytes, overflow checked.
    uchar *buf = fast_malloc(improc_scratch_mul((size_t)src->cols, (size_t)cn));
    improc_scratch_check((void *const[]){buf}, 1);

    for (int y = 0; y < rows; y++) {
        const int y0 = y - r < 0 ? 0 : y - r;
        const int y1 = (int32_t)y + r > rows - 1 ? rows - 1 : y + r;
        for (int i = 0; i < n; i++) {
            buf[i] = mat_row(src, y0)[i];
        }
        for (int yy = y0 + 1; yy <= y1; yy++) {
            vminmax_row(buf, mat_row(src, yy), n, is_dilate);
        }
        hminmax_row(buf, mat_row(dst, y), src->cols, cn, r, is_dilate);
    }

    fast_free(buf);
}

void erode(struct Mat *dst, const struct Mat *src, int ksize) {
    morph(dst, src, ksize, 0);
}

void dilate(struct Mat *dst, const struct Mat *src, int ksize) {
    morph(dst, src, ksize, 1);
}

static void morph_pair(struct Mat *dst, struct Mat *tmp, const struct Mat *src,
                       int ksize, int first_is_dilate) {
    ASSERT(dst != NULL && tmp != NULL && src != NULL);
    ASSERT(tmp != src && tmp != dst && tmp->data != src->data &&
           tmp->data != dst->data);
    ASSERT(mat_type(tmp) == mat_type(src));
    ASSERT(tmp->rows == src->rows && tmp->cols == src->cols);
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    morph(tmp, src, ksize, first_is_dilate);
    morph(dst, tmp, ksize, !first_is_dilate);
}

void morph_open(struct Mat *dst, struct Mat *tmp, const struct Mat *src,
                int ksize) {
    morph_pair(dst, tmp, src, ksize, 0);
}

void morph_close(struct Mat *dst, struct Mat *tmp, const struct Mat *src,
                 int ksize) {
    morph_pair(dst, tmp, src, ksize, 1);
}
