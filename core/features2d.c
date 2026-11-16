#include "features2d.h"

#include <string.h>

#include "improc_internal.h"
#include "utils.h"

// Radius-3 Bresenham circle as (dx, dy), in cv2 order: starts below the centre
// and turns clockwise in image coordinates.
static const signed char fast_circle[16][2] = {
    {0, 3},  {1, 3},   {2, 2},   {3, 1},   {3, 0},  {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3},
};

// Returns 1 when the 16-bit circular mask holds a run of 9 set bits.
static int has_run9(uint32_t mask) {
    uint32_t m = mask | (mask << 16); // unroll the circle once
    uint32_t r = m;
    for (int k = 1; k < 9; k++) {
        r &= m >> k;
    }
    return (r & 0xFFFFu) != 0;
}

// Reads the 16 circle differences v - p around (x, y). The caller keeps
// 3 <= x <= cols - 4 and 3 <= y <= rows - 4, so every read is inside src.
static void circle_diff(int d[16], const struct Mat *src, int x, int y) {
    const int p = mat_row(src, y)[x];
    for (int k = 0; k < 16; k++) {
        d[k] =
            (int)mat_row(src, y + fast_circle[k][1])[x + fast_circle[k][0]] - p;
    }
}

// Returns 1 when (x, y) is a FAST-9 corner for threshold t, and fills d.
static int is_corner(int d[16], const struct Mat *src, int x, int y, int t) {
    const int    p   = mat_row(src, y)[x];
    const uchar *up  = mat_row(src, y - 3);
    const uchar *dn  = mat_row(src, y + 3);
    const uchar *mid = mat_row(src, y);
    // Any 9-pixel arc covers at least 2 of the 4 compass pixels (0, 4, 8, 12).
    const int c[4]   = {dn[x], mid[x + 3], up[x], mid[x - 3]};
    int       bright = 0;
    int       dark   = 0;
    for (int k = 0; k < 4; k++) {
        bright += c[k] > p + t;
        dark += c[k] < p - t;
    }
    if (bright < 2 && dark < 2) {
        return 0;
    }
    circle_diff(d, src, x, y);
    uint32_t bmask = 0;
    uint32_t dmask = 0;
    for (int k = 0; k < 16; k++) {
        bmask |= (uint32_t)(d[k] > t) << k;
        dmask |= (uint32_t)(d[k] < -t) << k;
    }
    return has_run9(bmask) || has_run9(dmask);
}

// Largest threshold that still makes the point a corner, minus 1: the best
// arc of 9 by its weakest pixel, over both polarities.
static int corner_score(const int d[16]) {
    int best = 0;
    for (int s = 0; s < 16; s++) {
        int lo = 255;
        int hi = -255;
        for (int k = 0; k < 9; k++) {
            const int v = d[(s + k) & 15];
            lo          = v < lo ? v : lo;
            hi          = v > hi ? v : hi;
        }
        best = lo > best ? lo : best;
        best = -hi > best ? -hi : best;
    }
    return best - 1;
}

static void emit(struct KeyPoint *kps, int32_t capacity, int32_t *total, int x,
                 int y, int response) {
    if (*total < capacity) {
        kps[*total].x        = x;
        kps[*total].y        = y;
        kps[*total].response = response;
    }
    (*total)++;
}

int32_t fast_detect(struct KeyPoint *kps, int32_t capacity,
                    const struct Mat *src, int threshold, int nonmax) {
    ASSERT(src != NULL && src->data != NULL);
    ASSERT(CV_MAT_TYPE(src->flags) == CV_8UC1);
    ASSERT(threshold >= 0 && threshold <= 255);
    ASSERT(capacity >= 0);
    ASSERT(kps != NULL || capacity == 0);

    int32_t total = 0;
    if (src->rows < 7 || src->cols < 7) {
        return 0;
    }
    const int x_end = src->cols - 3; // exclusive
    const int y_end = src->rows - 3; // exclusive
    int       d[16];

    if (!nonmax) {
        for (int y = 3; y < y_end; y++) {
            for (int x = 3; x < x_end; x++) {
                if (is_corner(d, src, x, y, threshold)) {
                    emit(kps, capacity, &total, x, y, 0);
                }
            }
        }
        return total;
    }

    // Ring of 3 score rows; row y lives in slot y % 3. A score of 0 means
    // "not a corner"; a real corner of score 0 can never beat its neighbours
    // either, which is also what cv2 does.
    const size_t cols      = (size_t)src->cols;
    const size_t ring_size = improc_scratch_mul(3, cols);
    uchar       *ring      = fast_malloc(ring_size);
    improc_scratch_check((void *const[]){ring}, 1);
    memset(ring, 0, ring_size);

    // Row y_end is a flush row: it stays all 0 and lets row y_end - 1 emit.
    for (int y = 3; y <= y_end; y++) {
        uchar *cur = ring + (size_t)(y % 3) * cols;
        memset(cur, 0, cols);
        if (y < y_end) {
            for (int x = 3; x < x_end; x++) {
                if (is_corner(d, src, x, y, threshold)) {
                    cur[x] = (uchar)corner_score(d);
                }
            }
        }
        const int yc = y - 1; // candidate row, its rows yc - 1 .. y are scored
        if (yc < 3) {
            continue;
        }
        const uchar *pr = ring + (size_t)((yc - 1) % 3) * cols;
        const uchar *cr = ring + (size_t)(yc % 3) * cols;
        for (int x = 3; x < x_end; x++) {
            const int s = cr[x];
            if (s > pr[x - 1] && s > pr[x] && s > pr[x + 1] && s > cr[x - 1] &&
                s > cr[x + 1] && s > cur[x - 1] && s > cur[x] &&
                s > cur[x + 1]) {
                emit(kps, capacity, &total, x, yc, s);
            }
        }
    }
    fast_free(ring);
    return total;
}
