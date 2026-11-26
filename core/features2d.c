#include "features2d.h"

#include <stdlib.h>
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
        kps[*total].angle    = -1;
    }
    (*total)++;
}

static size_t  fast_ring_bytes(const struct Mat *src);
static int32_t fast_nonmax(struct KeyPoint *kps, int32_t capacity,
                           const struct Mat *src, int threshold, uchar *ring);

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

    uchar *ring = fast_malloc(fast_ring_bytes(src));
    improc_scratch_check((void *const[]){ring}, 1);
    total = fast_nonmax(kps, capacity, src, threshold, ring);
    fast_free(ring);
    return total;
}

// Score ring of fast_nonmax: 3 rows of cols bytes, overflow checked.
static size_t fast_ring_bytes(const struct Mat *src) {
    return improc_scratch_mul(3, (size_t)src->cols);
}

// FAST with non-max suppression on an image of at least 7x7, using a ring
// from the caller (fast_ring_bytes(src) bytes), so a caller that holds other
// scratch can allocate and check all of it together. Returns the total.
static int32_t fast_nonmax(struct KeyPoint *kps, int32_t capacity,
                           const struct Mat *src, int threshold, uchar *ring) {
    const int    x_end = src->cols - 3; // exclusive
    const int    y_end = src->rows - 3; // exclusive
    const size_t cols  = (size_t)src->cols;
    int32_t      total = 0;
    int          d[16];

    // Ring of 3 score rows; row y lives in slot y % 3. A score of 0 means
    // "not a corner"; a real corner of score 0 can never beat its neighbours
    // either, which is also what cv2 does.
    memset(ring, 0, 3 * cols);

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
    return total;
}

// ---- ORB --------------------------------------------------------------------

// Q14 cos and sin of step * 360 / ORB_ANGLE_STEPS degrees, rounded.
static const int16_t orb_cos_sin_q14[32][2] = {
    {16384, 0},       {16069, 3196},   {15137, 6270},   {13623, 9102},
    {11585, 11585},   {9102, 13623},   {6270, 15137},   {3196, 16069},
    {0, 16384},       {-3196, 16069},  {-6270, 15137},  {-9102, 13623},
    {-11585, 11585},  {-13623, 9102},  {-15137, 6270},  {-16069, 3196},
    {-16384, 0},      {-16069, -3196}, {-15137, -6270}, {-13623, -9102},
    {-11585, -11585}, {-9102, -13623}, {-6270, -15137}, {-3196, -16069},
    {0, -16384},      {3196, -16069},  {6270, -15137},  {9102, -13623},
    {11585, -11585},  {13623, -9102},  {15137, -6270},  {16069, -3196},
};

// 256 BRIEF test pairs (x1, y1, x2, y2), every coordinate in -13..13.
// tools/gen_orb_pattern.py generates them with integers only: xorshift32
// seeded with 0x9E3779B9; a coordinate is (sum of four (state >> 27)) - 62,
// divided by 3 and rounded half away from zero (about Gaussian, sigma 6.2);
// a pair is rejected when a coordinate is outside -13..13 or both points are
// the same. tests/test_features.c regenerates the table and compares it.
const signed char improc_orb_pattern[256][4] = {
    {-5, -4, 1, 12},    {3, -4, 0, 4},      {2, 7, 0, 6},
    {-4, 3, 0, -9},     {-2, -7, 1, 6},     {-8, 3, 0, 6},
    {4, -3, -5, 5},     {-9, 2, 3, 3},      {-12, -2, 2, 0},
    {5, -2, 5, 6},      {1, -9, -2, -1},    {-6, 5, -9, 4},
    {3, -11, 1, 0},     {-5, 2, 7, 4},      {-3, 2, -6, 10},
    {-9, -6, 8, 9},     {-4, 0, -10, -2},   {4, 1, -6, -3},
    {0, -4, 6, -11},    {-5, 0, 5, -3},     {8, -5, 13, 0},
    {5, -1, -1, 3},     {4, -11, -7, -3},   {7, -10, -2, -8},
    {6, 11, 4, 8},      {1, -5, 7, -4},     {6, -12, 13, 1},
    {13, -1, -8, -3},   {3, 0, -2, -8},     {9, 5, -11, -10},
    {11, -6, 0, -4},    {9, -4, -8, 4},     {5, 2, 4, -4},
    {-6, -5, 0, 3},     {-6, 2, 2, -2},     {7, 3, -1, 2},
    {8, 4, -5, 1},      {1, -2, -10, -1},   {6, 2, 6, 3},
    {3, 1, -1, 12},     {-3, 7, -4, -7},    {6, 7, 3, -6},
    {-2, 5, -8, -3},    {4, -5, -3, -6},    {2, 5, 2, -9},
    {-3, 4, -5, 2},     {-7, -2, 7, 11},    {7, -1, -7, -3},
    {6, 7, -5, -2},     {-2, -7, 0, 5},     {3, 13, -4, 13},
    {1, 3, -1, 10},     {-4, 1, 7, -9},     {6, -4, 6, -2},
    {2, 1, 7, 8},       {9, -6, -4, -5},    {6, 7, 3, 7},
    {8, 3, -7, 4},      {-3, -4, -2, 5},    {-7, -2, -10, 4},
    {-1, 6, 12, -9},    {3, -5, -3, -5},    {3, 1, 0, 7},
    {0, 9, 6, -7},      {10, -7, -2, -6},   {-3, -2, -4, 0},
    {0, 0, 6, 11},      {3, -5, -2, 2},     {2, -3, 0, -1},
    {-4, 8, 13, -2},    {1, -2, -8, -6},    {-2, 4, -2, 10},
    {6, 0, -4, 5},      {10, -5, -6, 1},    {-11, -1, 6, 5},
    {9, -8, -6, 5},     {1, -9, -3, -2},    {0, -1, -11, 10},
    {6, -8, -5, 10},    {1, -2, 0, -4},     {0, -5, 8, 11},
    {-3, -2, 4, -6},    {5, 9, 11, -4},     {0, 0, -2, -13},
    {-4, 5, -3, 2},     {-7, 2, -4, -4},    {-10, -8, 10, -7},
    {-5, -1, 6, 2},     {8, -1, -2, 9},     {-4, -7, -2, -7},
    {-3, -8, 6, -2},    {1, 6, 7, 4},       {-2, 0, 10, 3},
    {8, -1, -2, -1},    {-9, 2, 6, 1},      {-1, 2, 7, -9},
    {-12, 4, -1, -3},   {-5, 6, 3, 6},      {-5, -5, -4, 6},
    {-1, 8, -9, 4},     {3, -9, -1, -3},    {11, 5, 0, 7},
    {3, 3, 8, 3},       {1, -2, -2, -5},    {-4, 5, -8, -1},
    {-3, -7, 4, -7},    {5, -7, -2, -2},    {9, 13, -8, -8},
    {-5, 1, -13, 4},    {-1, 13, 7, -2},    {1, 10, -2, 1},
    {-9, -11, -5, -11}, {-13, 6, -3, 4},    {4, -3, 2, 4},
    {0, 9, -10, -3},    {-1, -2, 8, -11},   {-4, -6, 4, 8},
    {-1, 0, -2, -2},    {-10, -2, -10, -1}, {2, -1, 6, 1},
    {-10, -3, 0, -2},   {8, 3, -1, 12},     {-1, 1, -4, -3},
    {-1, 6, 8, -13},    {1, 6, -4, 0},      {-1, -5, 0, -4},
    {5, -4, 9, -6},     {3, 10, -8, -2},    {1, -1, 2, -5},
    {2, -3, 2, -5},     {1, -1, 9, -5},     {-10, -5, 10, -9},
    {-8, -9, 2, -1},    {11, 0, -5, -5},    {-9, 3, 2, -1},
    {8, 5, 1, 2},       {4, 2, 6, -5},      {-3, 1, 0, 7},
    {5, 1, 3, 5},       {3, 12, -2, -1},    {-9, -4, -1, 0},
    {-6, 3, -6, 1},     {6, -5, -9, 3},     {-8, -5, -4, 3},
    {3, -6, -7, 1},     {4, -8, -2, -1},    {-8, -3, 9, -10},
    {13, -4, -9, 4},    {-4, -13, 7, 2},    {-2, -2, -5, 1},
    {6, 10, -7, 8},     {-3, -5, 3, 4},     {4, -1, -4, 7},
    {-5, 6, 2, 1},      {8, -6, -8, 9},     {3, 12, -5, -4},
    {6, 3, 2, -1},      {3, 9, 9, -1},      {-5, 2, -3, -6},
    {5, 2, -6, -12},    {10, -8, -3, 3},    {0, -1, -11, -8},
    {0, 4, -4, 13},     {-2, 3, -5, 4},     {-11, 9, 2, 2},
    {-2, 5, 1, 1},      {-3, 3, 0, 10},     {10, 4, 2, -3},
    {0, -2, 4, 2},      {9, -5, 2, 6},      {3, -5, 0, -4},
    {0, 6, -11, 2},     {-4, 3, 12, 5},     {-7, 6, -4, -2},
    {-3, -3, 5, -6},    {-5, 4, 0, 2},      {-3, 2, -2, -12},
    {5, -2, 3, -6},     {-3, -3, -3, 9},    {-8, 4, -1, -3},
    {-7, -5, 2, -4},    {-5, 1, 7, -3},     {0, -6, 4, -9},
    {-9, -2, 3, -2},    {-11, 3, -2, 7},    {1, -3, -1, -8},
    {0, -6, -6, -7},    {6, -4, 1, -3},     {6, 9, -1, -2},
    {-3, -1, -5, 10},   {-8, -4, -4, -8},   {-7, 3, -1, -6},
    {1, -5, -5, 5},     {-1, 7, 3, 4},      {10, -2, -1, 8},
    {-10, 2, 13, 0},    {-1, 11, -5, -1},   {3, -11, 6, 0},
    {-4, -5, 3, -11},   {-9, -3, -8, 0},    {10, 6, 2, -3},
    {-3, -6, -2, -10},  {-2, -12, -6, -9},  {4, 4, 5, 8},
    {-8, -3, 6, 2},     {-3, 4, 1, -3},     {5, 12, -1, 2},
    {-12, -6, 3, 4},    {4, -3, -1, -6},    {1, -2, -5, 7},
    {0, -2, 2, -1},     {11, 11, -1, 3},    {-8, -9, -1, -8},
    {2, -1, 0, 1},      {8, 0, 1, 0},       {5, 7, 5, -4},
    {-7, -8, -5, 1},    {3, -13, -7, 0},    {13, 3, -10, 1},
    {7, 5, 3, 1},       {3, 3, -6, 2},      {-8, 6, 5, -6},
    {1, -8, 3, 10},     {-8, 4, 0, -4},     {6, 8, -12, -12},
    {5, 0, 8, -11},     {-3, -8, -13, 2},   {-8, -8, 9, 1},
    {-3, -2, -8, -10},  {2, 3, -2, 11},     {6, -3, 4, -3},
    {-3, 1, -2, 5},     {3, 10, 12, -3},    {3, -1, -3, -7},
    {7, 1, -13, 0},     {5, -2, -4, -3},    {-2, 8, -4, -11},
    {2, -4, -3, 2},     {1, -2, 9, -1},     {12, -7, -3, -1},
    {2, -1, -12, 2},    {1, -7, 10, -5},    {10, -6, -1, 4},
    {-9, -7, -3, 11},   {-4, -11, 0, -1},   {-3, 3, -9, -2},
    {2, 2, -10, 0},     {-1, -4, 3, -3},    {2, -6, -1, -12},
    {-1, -9, -11, 0},   {11, -1, -3, -3},   {-5, 10, -8, -8},
    {4, 2, 3, -1},      {-4, 4, -4, 7},     {-4, 5, -3, -3},
    {0, -6, 7, 10},
};

#define ORB_PATCH_R 15 // radius of the orientation disc

// Half widths of the radius-15 disc, row v = 0..15: round(sqrt(15^2 - v^2))
// up to v = 11, then the rows that make the disc symmetric under a 90-degree
// rotation (the construction of cv2's ORB). Point (u, v) is in the disc when
// |u| <= orb_umax[|v|], and then also |v| <= orb_umax[|u|]; plain rounding
// for every row would include (15, 3) but not (3, 15).
static const signed char orb_umax[ORB_PATCH_R + 1] = {
    15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3};

// Rounds a Q14 value to an integer, half away from zero. No right shift of a
// negative value.
static int round_q14(int32_t v) {
    return v >= 0 ? (int)((v + 8192) >> 14) : -(int)((8192 - v) >> 14);
}

// Harris measure 25 * det(M) - trace(M)^2 (k = 0.04 scaled by 25) of the 3x3
// Sobel gradients over the 7x7 window centred on (x, y). The caller keeps the
// window and its 1-pixel Sobel margin inside src.
int64_t improc_harris_measure(const struct Mat *src, int x, int y) {
    int32_t sxx = 0;
    int32_t syy = 0;
    int32_t sxy = 0;
    for (int v = y - 3; v <= y + 3; v++) {
        const uchar *a = mat_row(src, v - 1);
        const uchar *b = mat_row(src, v);
        const uchar *c = mat_row(src, v + 1);
        for (int u = x - 3; u <= x + 3; u++) {
            const int32_t gx =
                ((int32_t)a[u + 1] + 2 * (int32_t)b[u + 1] + c[u + 1]) -
                ((int32_t)a[u - 1] + 2 * (int32_t)b[u - 1] + c[u - 1]);
            const int32_t gy =
                ((int32_t)c[u - 1] + 2 * (int32_t)c[u] + c[u + 1]) -
                ((int32_t)a[u - 1] + 2 * (int32_t)a[u] + a[u + 1]);
            sxx += gx * gx; // |g| <= 1020: each sum <= 49 * 1020^2 < 2^31
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    const int64_t det = (int64_t)sxx * syy - (int64_t)sxy * sxy;
    const int64_t tr  = (int64_t)sxx + syy;
    return 25 * det - tr * tr;
}

// Orientation step closest to the intensity centroid (m10, m01) of the
// radius-15 disc: the step whose (cos, sin) has the largest dot product with
// it. Ties and a zero centroid give the lowest step.
int32_t improc_orb_angle(const struct Mat *src, int x, int y) {
    int32_t m10 = 0; // |m10|, |m01| <= 255 * 15 * 709 pixels < 2^22
    int32_t m01 = 0;
    for (int v = -ORB_PATCH_R; v <= ORB_PATCH_R; v++) {
        const uchar *row   = mat_row(src, y + v);
        const int    u_max = orb_umax[v < 0 ? -v : v];
        int32_t      sum   = 0;
        for (int u = -u_max; u <= u_max; u++) {
            sum += row[x + u];
            m10 += (int32_t)u * row[x + u];
        }
        m01 += (int32_t)v * sum;
    }
    int32_t best_step = 0;
    int64_t best      = 0;
    for (int s = 0; s < ORB_ANGLE_STEPS; s++) {
        const int64_t d = (int64_t)m10 * orb_cos_sin_q14[s][0] +
                          (int64_t)m01 * orb_cos_sin_q14[s][1];
        if (s == 0 || d > best) {
            best      = d;
            best_step = s;
        }
    }
    return best_step;
}

struct orb_rank {
    int64_t harris;
    int32_t index; // into the fast_detect output
};

// Highest Harris first; equal values keep raster order (lower index first).
static int orb_rank_cmp(const void *pa, const void *pb) {
    const struct orb_rank *a = pa;
    const struct orb_rank *b = pb;
    if (a->harris != b->harris) {
        return a->harris > b->harris ? -1 : 1;
    }
    return (a->index > b->index) - (a->index < b->index);
}

static int orb_inside(const struct KeyPoint *kp, const struct Mat *m) {
    return kp->x >= ORB_BORDER && kp->x < m->cols - ORB_BORDER &&
           kp->y >= ORB_BORDER && kp->y < m->rows - ORB_BORDER;
}

int32_t orb_detect(struct KeyPoint *kps, int32_t max_kps, const struct Mat *src,
                   int threshold) {
    ASSERT(src != NULL && src->data != NULL);
    ASSERT(CV_MAT_TYPE(src->flags) == CV_8UC1);
    ASSERT(threshold >= 0 && threshold <= 255);
    ASSERT(max_kps >= 0);
    ASSERT(kps != NULL || max_kps == 0);
    if (max_kps == 0 || src->rows <= 2 * ORB_BORDER ||
        src->cols <= 2 * ORB_BORDER) {
        return 0;
    }
    const int32_t n = fast_detect(NULL, 0, src, threshold, 1);
    if (n == 0) {
        return 0;
    }
    ASSERT((uintmax_t)n <= (uintmax_t)SIZE_MAX); // size_t is 16 bits on AVR
    // All sizes first (each may ASSERT on overflow), then all blocks, then one
    // check: a failure frees every block before it ASSERTs.
    const size_t un         = (size_t)n;
    const size_t all_bytes  = improc_scratch_mul(un, sizeof(struct KeyPoint));
    const size_t rank_bytes = improc_scratch_mul(un, sizeof(struct orb_rank));
    const size_t ring_bytes = fast_ring_bytes(src);
    uchar       *ring       = fast_malloc(ring_bytes);
    struct KeyPoint *all    = fast_malloc(all_bytes);
    struct orb_rank *rank   = fast_malloc(rank_bytes);
    improc_scratch_check((void *const[]){ring, all, rank}, 3);
    fast_nonmax(all, n, src, threshold, ring);
    fast_free(ring);

    int32_t m = 0;
    for (int32_t i = 0; i < n; i++) {
        if (orb_inside(&all[i], src)) {
            rank[m].harris = all[i].response;
            rank[m].index  = i;
            m++;
        }
    }
    // Stage 1: the best 2 * max_kps by FAST score (ties: raster order).
    qsort(rank, (size_t)m, sizeof *rank, orb_rank_cmp);
    if (m - max_kps > max_kps) { // m > 2 * max_kps without overflow
        m = 2 * max_kps;
    }
    for (int32_t i = 0; i < m; i++) {
        rank[i].harris = improc_harris_measure(src, all[rank[i].index].x,
                                               all[rank[i].index].y);
    }
    // Stage 2: rank those by Harris.
    qsort(rank, (size_t)m, sizeof *rank, orb_rank_cmp);
    const int32_t kept = m < max_kps ? m : max_kps;
    for (int32_t i = 0; i < kept; i++) {
        kps[i]       = all[rank[i].index];
        kps[i].angle = improc_orb_angle(src, kps[i].x, kps[i].y);
    }
    fast_free(rank);
    fast_free(all);
    return kept;
}

void orb_compute(uint8_t (*desc)[ORB_DESC_BYTES], const struct KeyPoint *kps,
                 int32_t n, const struct Mat *smooth) {
    ASSERT(smooth != NULL && smooth->data != NULL);
    ASSERT(CV_MAT_TYPE(smooth->flags) == CV_8UC1);
    ASSERT(n >= 0);
    ASSERT(n == 0 || (desc != NULL && kps != NULL));
    for (int32_t i = 0; i < n; i++) {
        const struct KeyPoint *kp = &kps[i];
        ASSERT(orb_inside(kp, smooth));
        ASSERT(kp->angle >= 0 && kp->angle < ORB_ANGLE_STEPS);
        const int32_t c = orb_cos_sin_q14[kp->angle][0];
        const int32_t s = orb_cos_sin_q14[kp->angle][1];
        memset(desc[i], 0, ORB_DESC_BYTES);
        for (int k = 0; k < 256; k++) {
            const signed char *p = improc_orb_pattern[k];
            // |rotated coordinate| <= 13 * sqrt(2) < ORB_BORDER
            const int   x1 = round_q14(p[0] * c - p[1] * s);
            const int   y1 = round_q14(p[0] * s + p[1] * c);
            const int   x2 = round_q14(p[2] * c - p[3] * s);
            const int   y2 = round_q14(p[2] * s + p[3] * c);
            const uchar v1 = mat_row(smooth, kp->y + y1)[kp->x + x1];
            const uchar v2 = mat_row(smooth, kp->y + y2)[kp->x + x2];
            if (v1 < v2) {
                desc[i][k >> 3] |= (uint8_t)(1u << (k & 7));
            }
        }
    }
}

int hamming_distance(const uint8_t a[ORB_DESC_BYTES],
                     const uint8_t b[ORB_DESC_BYTES]) {
    static const uint8_t nibble_bits[16] = {0, 1, 1, 2, 1, 2, 2, 3,
                                            1, 2, 2, 3, 2, 3, 3, 4};
    int                  d               = 0;
    for (int i = 0; i < ORB_DESC_BYTES; i++) {
        const unsigned x = (unsigned)(a[i] ^ b[i]);
        d += nibble_bits[x & 15u] + nibble_bits[x >> 4];
    }
    return d;
}

// Index of the descriptor in set[0..n) closest to d; ties: lowest index.
static int32_t closest(const uint8_t d[ORB_DESC_BYTES],
                       const uint8_t (*set)[ORB_DESC_BYTES], int32_t n,
                       int *dist) {
    int32_t best   = 0;
    int     best_d = hamming_distance(d, set[0]);
    for (int32_t j = 1; j < n; j++) {
        const int dj = hamming_distance(d, set[j]);
        if (dj < best_d) {
            best_d = dj;
            best   = j;
        }
    }
    *dist = best_d;
    return best;
}

int32_t match_descriptors(struct DMatch *matches,
                          const uint8_t (*query)[ORB_DESC_BYTES], int32_t nq,
                          const uint8_t (*train)[ORB_DESC_BYTES], int32_t nt,
                          int cross_check) {
    ASSERT(nq >= 0 && nt >= 0);
    ASSERT(nq == 0 || (query != NULL && matches != NULL));
    ASSERT(nt == 0 || train != NULL);
    if (nq == 0 || nt == 0) {
        return 0;
    }
    int32_t *back = NULL; // best query of each train descriptor
    if (cross_check) {
        ASSERT((uintmax_t)nt <= (uintmax_t)SIZE_MAX);
        back = fast_malloc(improc_scratch_mul((size_t)nt, sizeof *back));
        improc_scratch_check((void *const[]){back}, 1);
        int unused;
        for (int32_t t = 0; t < nt; t++) {
            back[t] = closest(train[t], query, nq, &unused);
        }
    }
    int32_t count = 0;
    for (int32_t q = 0; q < nq; q++) {
        int           d;
        const int32_t t = closest(query[q], train, nt, &d);
        if (back == NULL || back[t] == q) {
            matches[count].query    = q;
            matches[count].train    = t;
            matches[count].distance = d;
            count++;
        }
    }
    fast_free(back);
    return count;
}
