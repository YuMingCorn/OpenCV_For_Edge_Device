#include "filter.h"

#include <stdint.h>

#include "improc_internal.h"
#include "utils.h"

// Largest rows or cols the filters accept. Keeps every int32_t coordinate
// expression below (y + r + 1, cols + 2r) inside int32_t. On a target with a
// 16-bit int the Mat dimensions are already below it.
#define FILTER_MAX_DIM ((int32_t)(INT32_MAX / 4))

// Reflect 101 of p into [0, len). p stays within a few pixels of the image, so
// one reflection is the usual case. The arithmetic is int32_t, so it is exact
// on a 16-bit int as well; check_filter_dims keeps 2 * (len - 1) in range.
static inline int border_reflect_101(int32_t p, int32_t len) {
    if (len == 1) {
        return 0;
    }
    while (p < 0 || p >= len) {
        p = p < 0 ? -p : 2 * (len - 1) - p;
    }
    return (int)p;
}

static void check_filter_dims(const struct Mat *src) {
    ASSERT((int32_t)src->rows <= FILTER_MAX_DIM &&
           (int32_t)src->cols <= FILTER_MAX_DIM);
}

static int is_u8_image_type(const struct Mat *m) {
    size_t t = mat_type(m);
    return t == (size_t)CV_8UC1 || t == (size_t)CV_8UC3 || t == (size_t)CV_8UC4;
}

// Copies one row into pad, extending r pixels on each side with reflect 101.
// pad holds (cols + 2r) * cn bytes.
static void pad_row(const uchar *restrict s, uchar *restrict pad, int cols,
                    int cn, int r) {
    uchar *q = pad;
    for (int32_t x = -r; x < (int32_t)cols + r; x++, q += cn) {
        const uchar *p = s + (size_t)border_reflect_101(x, cols) * (size_t)cn;
        for (int c = 0; c < cn; c++) {
            q[c] = p[c];
        }
    }
}

// The horizontal pass keeps a ring of nslots = min(window, rows) rows, and
// row y lives in slot y % nslots. Output row y reads only rows in
// [max(0, y - r), min(rows - 1, y + r)], at most `window` consecutive rows,
// so a slot is never reused while its row is still needed.
static inline size_t ring_slot(int row, int nslots, size_t n) {
    return (size_t)(row % nslots) * n;
}

// Box horizontal pass: sliding window sum of ksize pixels per channel.
// pad holds (width + ksize - 1) * cn bytes, out holds width * cn.
static void box_hrow(const uchar *restrict pad, uint16_t *restrict out,
                     int width, int cn, int ksize) {
    for (int c = 0; c < cn; c++) {
        unsigned sum = 0;
        for (size_t i = 0; i < (size_t)ksize; i++) {
            sum += pad[i * (size_t)cn + (size_t)c];
        }
        out[c] = (uint16_t)sum;
        for (size_t x = 1; x < (size_t)width; x++) {
            sum += pad[(x + (size_t)ksize - 1) * (size_t)cn + (size_t)c];
            sum -= pad[(x - 1) * (size_t)cn + (size_t)c];
            out[x * (size_t)cn + (size_t)c] = (uint16_t)sum;
        }
    }
}

static void colsum_add(uint32_t *restrict sum, const uint16_t *restrict t,
                       size_t n) {
    for (size_t i = 0; i < n; i++) {
        sum[i] += t[i];
    }
}

static void colsum_sub(uint32_t *restrict sum, const uint16_t *restrict t,
                       size_t n) {
    for (size_t i = 0; i < n; i++) {
        sum[i] -= t[i];
    }
}

static void box_vrow(const uint32_t *restrict sum, uchar *restrict d, size_t n,
                     uint32_t area) {
    for (size_t i = 0; i < n; i++) {
        d[i] = (uchar)((sum[i] + area / 2) / area);
    }
}

void box_filter(struct Mat *dst, const struct Mat *src, int ksize) {
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(is_u8_image_type(src));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(ksize >= 1 && ksize <= 255 && (ksize & 1));
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    check_filter_dims(src);

    const int    rows   = src->rows;
    const int    cn     = (int)mat_channel_num(src);
    const int    n      = src->cols * cn;
    const int    r      = ksize / 2;
    const int    nslots = ksize < rows ? ksize : rows;
    const size_t un     = (size_t)n;

    // Scratch: ring ksize*n*2 + pad (cols+2r)*cn + sum n*4, overflow checked.
    const size_t ring_bytes = improc_scratch_mul(
        improc_scratch_mul((size_t)nslots, un), sizeof(uint16_t));
    const size_t pad_bytes = improc_scratch_mul(
        improc_scratch_add((size_t)src->cols, 2 * (size_t)r), (size_t)cn);
    const size_t sum_bytes = improc_scratch_mul(un, sizeof(uint32_t));
    uint16_t    *ring      = fast_malloc(ring_bytes);
    uchar       *pad       = fast_malloc(pad_bytes);
    uint32_t    *sum       = fast_malloc(sum_bytes);
    improc_scratch_check((void *const[]){ring, pad, sum}, 3);

    // `next` is the first row whose horizontal pass is not computed yet.
    int next = 0;
    for (; next <= (r < rows - 1 ? r : rows - 1); next++) {
        pad_row(mat_row(src, next), pad, src->cols, cn, r);
        box_hrow(pad, ring + ring_slot(next, nslots, un), src->cols, cn, ksize);
    }
    for (size_t i = 0; i < un; i++) {
        sum[i] = 0;
    }
    for (int i = -r; i <= r; i++) {
        colsum_add(
            sum, ring + ring_slot(border_reflect_101(i, rows), nslots, un), un);
    }
    const uint32_t area = (uint32_t)ksize * (uint32_t)ksize;
    for (int y = 0; y < rows; y++) {
        box_vrow(sum, mat_row(dst, y), un, area);
        // Subtract the outgoing row before its slot can be overwritten.
        colsum_sub(
            sum, ring + ring_slot(border_reflect_101(y - r, rows), nslots, un),
            un);
        const int32_t in  = (int32_t)y + r + 1;
        const int     top = in < rows - 1 ? (int)in : rows - 1;
        for (; next <= top; next++) {
            pad_row(mat_row(src, next), pad, src->cols, cn, r);
            box_hrow(pad, ring + ring_slot(next, nslots, un), src->cols, cn,
                     ksize);
        }
        colsum_add(sum,
                   ring + ring_slot(border_reflect_101(in, rows), nslots, un),
                   un);
    }

    fast_free(sum);
    fast_free(pad);
    fast_free(ring);
}

static const int gauss_weights[4][7] = {
    {256, 0, 0, 0, 0, 0, 0},
    {64, 128, 64, 0, 0, 0, 0},
    {16, 64, 96, 64, 16, 0, 0},
    {8, 28, 56, 72, 56, 28, 8},
};

// Weighted horizontal pass. Largest sum is 256 * 255, which fits uint16_t.
static void gauss_hrow(const uchar *restrict pad, uint16_t *restrict out,
                       int width, int cn, const int *w, int ksize) {
    size_t n = (size_t)width * (size_t)cn;
    for (size_t i = 0; i < n; i++) {
        unsigned acc = 0;
        for (size_t k = 0; k < (size_t)ksize; k++) {
            acc += (unsigned)w[k] * pad[i + k * (size_t)cn];
        }
        out[i] = (uint16_t)acc;
    }
}

static void gauss_vrow(const uint16_t *const *restrict rows_in,
                       uchar *restrict d, size_t n, const int *w, int ksize) {
    for (size_t i = 0; i < n; i++) {
        uint32_t acc = 0;
        for (int k = 0; k < ksize; k++) {
            acc += (uint32_t)w[k] * rows_in[k][i];
        }
        d[i] = (uchar)((acc + 32768u) >> 16);
    }
}

void gaussian_blur(struct Mat *dst, const struct Mat *src, int ksize) {
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(is_u8_image_type(src));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7);
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    check_filter_dims(src);

    const int    rows   = src->rows;
    const int    cn     = (int)mat_channel_num(src);
    const int    n      = src->cols * cn;
    const int    r      = ksize / 2;
    const int   *w      = gauss_weights[r];
    const int    nslots = ksize < rows ? ksize : rows;
    const size_t un     = (size_t)n;

    // Scratch: ring ksize*n*2 + pad (cols+2r)*cn, overflow checked.
    const size_t ring_bytes = improc_scratch_mul(
        improc_scratch_mul((size_t)nslots, un), sizeof(uint16_t));
    const size_t pad_bytes = improc_scratch_mul(
        improc_scratch_add((size_t)src->cols, 2 * (size_t)r), (size_t)cn);
    uint16_t *ring = fast_malloc(ring_bytes);
    uchar    *pad  = fast_malloc(pad_bytes);
    improc_scratch_check((void *const[]){ring, pad}, 2);

    int next = 0;
    for (int y = 0; y < rows; y++) {
        const int top = (int32_t)y + r < (int32_t)rows - 1 ? y + r : rows - 1;
        for (; next <= top; next++) {
            pad_row(mat_row(src, next), pad, src->cols, cn, r);
            gauss_hrow(pad, ring + ring_slot(next, nslots, un), src->cols, cn,
                       w, ksize);
        }
        const uint16_t *in[7];
        for (int k = 0; k < ksize; k++) {
            in[k] =
                ring + ring_slot(border_reflect_101((int32_t)y + k - r, rows),
                                 nslots, un);
        }
        gauss_vrow(in, mat_row(dst, y), un, w, ksize);
    }

    fast_free(pad);
    fast_free(ring);
}

// Sobel horizontal pass with a 3-tap kernel; |result| <= 1020 fits int16_t.
static void sobel_hrow(const uchar *restrict pad, int16_t *restrict out,
                       size_t n, size_t cn, const int *w) {
    for (size_t i = 0; i < n; i++) {
        out[i] = (int16_t)(w[0] * pad[i] + w[1] * pad[i + cn] +
                           w[2] * pad[i + 2 * cn]);
    }
}

static void sobel_vrow(const int16_t *restrict t0, const int16_t *restrict t1,
                       const int16_t *restrict t2, int16_t *restrict d,
                       size_t n, const int *w) {
    for (size_t i = 0; i < n; i++) {
        d[i] = (int16_t)(w[0] * t0[i] + w[1] * t1[i] + w[2] * t2[i]);
    }
}

void sobel(struct Mat *dst, const struct Mat *src, int dx, int dy) {
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(mat_type(src) == (size_t)CV_8UC1);
    ASSERT(mat_type(dst) == (size_t)CV_16SC1);
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT((dx == 1 && dy == 0) || (dx == 0 && dy == 1));
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    check_filter_dims(src);

    static const int diff[3]   = {-1, 0, 1};
    static const int smooth[3] = {1, 2, 1};
    const int       *wh        = dx ? diff : smooth;
    const int       *wv        = dx ? smooth : diff;

    const int    rows   = src->rows;
    const int    cols   = src->cols;
    const int    nslots = 3 < rows ? 3 : rows;
    const size_t uc     = (size_t)cols;

    // Scratch: ring 3*cols*2 + pad cols+2, overflow checked.
    const size_t ring_bytes = improc_scratch_mul(
        improc_scratch_mul((size_t)nslots, uc), sizeof(int16_t));
    const size_t pad_bytes = improc_scratch_add(uc, 2);
    int16_t     *ring      = fast_malloc(ring_bytes);
    uchar       *pad       = fast_malloc(pad_bytes);
    improc_scratch_check((void *const[]){ring, pad}, 2);

    int next = 0;
    for (int y = 0; y < rows; y++) {
        const int top = (int32_t)y + 1 < (int32_t)rows - 1 ? y + 1 : rows - 1;
        for (; next <= top; next++) {
            pad_row(mat_row(src, next), pad, cols, 1, 1);
            sobel_hrow(pad, ring + ring_slot(next, nslots, uc), uc, 1, wh);
        }
        const int16_t *t0 =
            ring + ring_slot(border_reflect_101(y - 1, rows), nslots, uc);
        const int16_t *t1 = ring + ring_slot(y, nslots, uc);
        const int16_t *t2 =
            ring +
            ring_slot(border_reflect_101((int32_t)y + 1, rows), nslots, uc);
        sobel_vrow(t0, t1, t2, (int16_t *)mat_row(dst, y), uc, wv);
    }

    fast_free(pad);
    fast_free(ring);
}

// ---- median_blur ------------------------------------------------------------

// Largest ksize: the window holds at most 255 * 255 = 65025 values, so every
// histogram bin fits uint16_t.
#define MEDIAN_MAX_KSIZE 255

static inline int clamp_index(int32_t p, int32_t len) {
    return (int)(p < 0 ? 0 : (p >= len ? len - 1 : p));
}

// Compare-exchange: afterwards a <= b.
#define MEDIAN_SORT2(a, b)                                                     \
    do {                                                                       \
        if ((a) > (b)) {                                                       \
            uchar t_ = (a);                                                    \
            (a)      = (b);                                                    \
            (b)      = t_;                                                     \
        }                                                                      \
    } while (0)

// Median of 9 values with a 19-exchange sorting network.
static inline uchar median_of_9(uchar p[9]) {
    MEDIAN_SORT2(p[1], p[2]);
    MEDIAN_SORT2(p[4], p[5]);
    MEDIAN_SORT2(p[7], p[8]);
    MEDIAN_SORT2(p[0], p[1]);
    MEDIAN_SORT2(p[3], p[4]);
    MEDIAN_SORT2(p[6], p[7]);
    MEDIAN_SORT2(p[1], p[2]);
    MEDIAN_SORT2(p[4], p[5]);
    MEDIAN_SORT2(p[7], p[8]);
    MEDIAN_SORT2(p[0], p[3]);
    MEDIAN_SORT2(p[5], p[8]);
    MEDIAN_SORT2(p[4], p[7]);
    MEDIAN_SORT2(p[3], p[6]);
    MEDIAN_SORT2(p[1], p[4]);
    MEDIAN_SORT2(p[2], p[5]);
    MEDIAN_SORT2(p[4], p[7]);
    MEDIAN_SORT2(p[4], p[2]);
    MEDIAN_SORT2(p[6], p[4]);
    MEDIAN_SORT2(p[4], p[2]);
    return p[4];
}

// Wirth selection: returns the element of rank k (0-based) of a[0..n-1] and
// reorders a. Used for the 25 values of a 5x5 window.
static uchar median_select(uchar *a, int n, int k) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        uchar pivot = a[k];
        int   i     = lo;
        int   j     = hi;
        do {
            while (a[i] < pivot) {
                i++;
            }
            while (pivot < a[j]) {
                j--;
            }
            if (i <= j) {
                uchar t = a[i];
                a[i]    = a[j];
                a[j]    = t;
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) {
            lo = i;
        }
        if (k < i) {
            hi = j;
        }
    }
    return a[k];
}

// ksize 3 or 5: gathers each window (replicate border) and selects. rp holds
// the ksize source rows, already clamped to the image.
static void median_small_row(const uchar *const *rp, uchar *restrict d,
                             int cols, int cn, int ksize) {
    const int r = ksize / 2;
    for (int x = 0; x < cols; x++) {
        // Byte offsets as size_t: cols * cn may exceed a 16-bit int.
        size_t xs[5];
        for (int j = 0; j < ksize; j++) {
            xs[j] = (size_t)clamp_index((int32_t)x + j - r, cols) * (size_t)cn;
        }
        for (int c = 0; c < cn; c++) {
            uchar w[25];
            int   n = 0;
            for (int i = 0; i < ksize; i++) {
                for (int j = 0; j < ksize; j++) {
                    w[n++] = rp[i][xs[j] + (size_t)c];
                }
            }
            d[(size_t)x * (size_t)cn + (size_t)c] =
                ksize == 3 ? median_of_9(w) : median_select(w, 25, 12);
        }
    }
}

// ksize >= 7: Huang sliding-window histogram along one row for one channel.
// hist counts the window values. med is the current median and below is the
// number of window values smaller than med, so the median is the smallest
// value m with below(m) + hist[m] > half, half = ksize^2 / 2. Moving the
// window by one pixel changes ksize values on each side, and med then moves by
// the few bins the change needs.
static void median_huang_row(const uchar *const *rp, uchar *restrict d,
                             int cols, int cn, int c, int ksize,
                             uint16_t *hist) {
    const int     r    = ksize / 2;
    const int32_t half = ((int32_t)ksize * (int32_t)ksize) / 2;
    const size_t  ucn  = (size_t)cn;
    const size_t  uc   = (size_t)c;

    for (int v = 0; v < 256; v++) {
        hist[v] = 0;
    }
    for (int j = -r; j <= r; j++) {
        const size_t off = (size_t)clamp_index(j, cols) * ucn + uc;
        for (int i = 0; i < ksize; i++) {
            hist[rp[i][off]]++;
        }
    }
    int     med   = 0;
    int32_t below = 0;
    for (int x = 0;; x++) {
        while (below + (int32_t)hist[med] <= half) {
            below += hist[med];
            med++;
        }
        while (below > half) {
            med--;
            below -= hist[med];
        }
        d[(size_t)x * ucn + uc] = (uchar)med;
        if (x + 1 >= cols) {
            break;
        }
        // Column leaving, then column entering (never above ksize^2 values).
        const size_t out_off =
            (size_t)clamp_index((int32_t)x - r, cols) * ucn + uc;
        const size_t in_off =
            (size_t)clamp_index((int32_t)x + 1 + r, cols) * ucn + uc;
        for (int i = 0; i < ksize; i++) {
            const int v = rp[i][out_off];
            hist[v]--;
            if (v < med) {
                below--;
            }
        }
        for (int i = 0; i < ksize; i++) {
            const int v = rp[i][in_off];
            hist[v]++;
            if (v < med) {
                below++;
            }
        }
    }
}

void median_blur(struct Mat *dst, const struct Mat *src, int ksize) {
    ASSERT(dst != src && dst->data != src->data);
    ASSERT(is_u8_image_type(src));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(ksize >= 3 && ksize <= MEDIAN_MAX_KSIZE && (ksize & 1));
    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    check_filter_dims(src);

    const int rows = src->rows;
    const int cols = src->cols;
    const int cn   = (int)mat_channel_num(src);
    const int r    = ksize / 2;

    // Scratch: ksize <= 5 needs none (rows go in a small stack array).
    // Otherwise ksize row pointers + a 256-bin uint16_t histogram.
    const uchar **rp   = NULL;
    uint16_t     *hist = NULL;
    const uchar  *small_rp[5];
    if (ksize > 5) {
        const size_t rp_bytes =
            improc_scratch_mul((size_t)ksize, sizeof(const uchar *));
        const size_t hist_bytes = improc_scratch_mul(256, sizeof(uint16_t));
        rp                      = fast_malloc(rp_bytes);
        hist                    = fast_malloc(hist_bytes);
        improc_scratch_check((void *const[]){rp, hist}, 2);
    }
    const uchar **rows_in = ksize > 5 ? rp : small_rp;

    for (int y = 0; y < rows; y++) {
        for (int i = 0; i < ksize; i++) {
            rows_in[i] = mat_row(src, clamp_index((int32_t)y + i - r, rows));
        }
        uchar *d = mat_row(dst, y);
        if (ksize <= 5) {
            median_small_row(rows_in, d, cols, cn, ksize);
        } else {
            for (int c = 0; c < cn; c++) {
                median_huang_row(rows_in, d, cols, cn, c, ksize, hist);
            }
        }
    }

    fast_free(hist);
    fast_free(rp);
}
