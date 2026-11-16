#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "improc.h"
#include "improc_internal.h"
#include "test_alloc_util.h"
#include "test_expect_assert.h"
#include "test_mat_util.h"

// ---- Naive reference -------------------------------------------------------
// Written without the library's bit masks or score formula: each arc is
// checked pixel by pixel, and the score is found by raising the threshold
// until the point stops being a corner.

static const int ref_circle[16][2] = {
    {0, 3},  {1, 3},   {2, 2},   {3, 1},   {3, 0},  {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3},
};

static int ref_px(const struct Mat *m, int x, int y) {
    return mat_row(m, y)[x];
}

static int ref_is_corner(const struct Mat *m, int x, int y, int t) {
    const int p = ref_px(m, x, y);
    for (int s = 0; s < 16; s++) {
        int all_bright = 1;
        int all_dark   = 1;
        for (int k = 0; k < 9; k++) {
            const int *o = ref_circle[(s + k) % 16];
            const int  v = ref_px(m, x + o[0], y + o[1]);
            all_bright &= v > p + t;
            all_dark &= v < p - t;
        }
        if (all_bright || all_dark) {
            return 1;
        }
    }
    return 0;
}

static int ref_score(const struct Mat *m, int x, int y, int t) {
    int s = t;
    while (s < 255 && ref_is_corner(m, x, y, s + 1)) {
        s++;
    }
    return s; // largest threshold that still gives a corner
}

static int ref_in_range(const struct Mat *m, int x, int y) {
    return x >= 3 && y >= 3 && x <= m->cols - 4 && y <= m->rows - 4;
}

// Score map: score of each corner, 0 elsewhere (and outside the tested area).
static int ref_map(const struct Mat *m, int x, int y, int t) {
    if (x < 0 || y < 0 || x >= m->cols || y >= m->rows ||
        !ref_in_range(m, x, y) || !ref_is_corner(m, x, y, t)) {
        return 0;
    }
    return ref_score(m, x, y, t);
}

// Fills out in raster order and returns the count (no capacity limit).
static int32_t ref_fast(struct KeyPoint *out, const struct Mat *m, int t,
                        int nonmax) {
    int32_t n = 0;
    for (int y = 3; y <= m->rows - 4; y++) {
        for (int x = 3; x <= m->cols - 4; x++) {
            if (!ref_is_corner(m, x, y, t)) {
                continue;
            }
            int s = 0;
            if (nonmax) {
                s        = ref_score(m, x, y, t);
                int keep = 1;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if ((dx || dy) &&
                            !(s > ref_map(m, x + dx, y + dy, t))) {
                            keep = 0;
                        }
                    }
                }
                if (!keep) {
                    continue;
                }
            }
            out[n].x        = x;
            out[n].y        = y;
            out[n].response = s;
            n++;
        }
    }
    return n;
}

static uint32_t xs_state = 2463534242u;

static uint32_t xorshift32(void) {
    xs_state ^= xs_state << 13;
    xs_state ^= xs_state >> 17;
    xs_state ^= xs_state << 5;
    return xs_state;
}

// Random image; with block > 1, constant squares of that size so that
// neighbouring corners get equal scores (NMS ties).
static void fill_random(struct Mat *m, int block) {
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->cols; x++) {
            mat_row(m, y)[x] = (uchar)(xorshift32() >> 24);
        }
    }
    if (block > 1) {
        for (int y = 0; y < m->rows; y++) {
            for (int x = 0; x < m->cols; x++) {
                mat_row(m, y)[x] = mat_row(m, y - y % block)[x - x % block];
            }
        }
    }
}

static void check_same(const struct KeyPoint *got, int32_t ngot,
                       const struct KeyPoint *want, int32_t nwant) {
    CHECK_EQ_INT(ngot, nwant);
    for (int32_t i = 0; i < ngot && i < nwant; i++) {
        if (got[i].x != want[i].x || got[i].y != want[i].y ||
            got[i].response != want[i].response) {
            TEST_FAIL_("point %ld: (%ld, %ld, %ld), expected (%ld, %ld, %ld)",
                       (long)i, (long)got[i].x, (long)got[i].y,
                       (long)got[i].response, (long)want[i].x, (long)want[i].y,
                       (long)want[i].response);
            return;
        }
    }
}

// ---- Hand-computed cases ---------------------------------------------------

// 7x7, centre 100, circle pixels 0..count-1 (from start) set to v, others 100.
static struct Mat *arc_image(int start, int count, int v) {
    struct Mat *m = alloc_mat(7, 7, CV_8UC1);
    memset(m->data, 100, (size_t)7 * m->step[0]);
    for (int k = 0; k < count; k++) {
        const int *o                   = ref_circle[(start + k) % 16];
        mat_row(m, 3 + o[1])[3 + o[0]] = (uchar)v;
    }
    return m;
}

static void test_fast_bright_arc(void) {
    // 9 pixels at 200: corner for t < 100, score 99.
    struct Mat     *m = arc_image(5, 9, 200);
    struct KeyPoint kp[2];
    CHECK_EQ_INT(fast_detect(kp, 2, m, 99, 0), 1);
    CHECK_EQ_INT(kp[0].x, 3);
    CHECK_EQ_INT(kp[0].y, 3);
    CHECK_EQ_INT(kp[0].response, 0);
    CHECK_EQ_INT(fast_detect(kp, 2, m, 100, 0), 0); // strict compare
    CHECK_EQ_INT(fast_detect(kp, 2, m, 20, 1), 1);
    CHECK_EQ_INT(kp[0].response, 99);
    free_mat(&m);
}

static void test_fast_dark_arc_wraps(void) {
    // 9 pixels at 30 that wrap around index 15 -> 0: score 69.
    struct Mat     *m = arc_image(12, 9, 30);
    struct KeyPoint kp[1];
    CHECK_EQ_INT(fast_detect(kp, 1, m, 10, 1), 1);
    CHECK_EQ_INT(kp[0].response, 69);
    CHECK_EQ_INT(fast_detect(kp, 1, m, 70, 1), 0);
    free_mat(&m);
}

static void test_fast_arc_of_8_is_not_a_corner(void) {
    struct Mat     *m = arc_image(3, 8, 255);
    struct KeyPoint kp[1];
    CHECK_EQ_INT(fast_detect(kp, 1, m, 0, 0), 0);
    CHECK_EQ_INT(fast_detect(kp, 1, m, 0, 1), 0);
    free_mat(&m);
}

static void test_fast_score_uses_weakest_pixel(void) {
    // 10 bright pixels, one of them weaker: the best 9-arc skips it.
    struct Mat *m                  = arc_image(0, 10, 200);
    const int  *o                  = ref_circle[0];
    mat_row(m, 3 + o[1])[3 + o[0]] = 150;
    struct KeyPoint kp[1];
    CHECK_EQ_INT(fast_detect(kp, 1, m, 10, 1), 1);
    CHECK_EQ_INT(kp[0].response, 99); // arc 1..9, all 200
    free_mat(&m);
}

static void test_fast_score_one_corner(void) {
    // Arc of 9 at 102 around 100, threshold 0: score 1. Its neighbours lie
    // outside the tested area, so they must count as 0 (zeroed score ring).
    struct Mat     *m = arc_image(0, 9, 102);
    struct KeyPoint kp[1];
    CHECK_EQ_INT(fast_detect(kp, 1, m, 0, 1), 1);
    CHECK_EQ_INT(kp[0].response, 1);
    CHECK_EQ_INT(kp[0].angle, -1);
    free_mat(&m);
}

static void test_fast_small_images(void) {
    struct KeyPoint kp[1];
    for (int n = 0; n < 7; n++) {
        struct Mat *m = alloc_mat(n, 20, CV_8UC1);
        struct Mat *w = alloc_mat(20, n, CV_8UC1);
        fill_random(m, 1);
        fill_random(w, 1);
        CHECK_EQ_INT(fast_detect(kp, 1, m, 0, 1), 0);
        CHECK_EQ_INT(fast_detect(kp, 1, w, 0, 0), 0);
        free_mat(&m);
        free_mat(&w);
    }
}

// ---- Sweeps against the reference ------------------------------------------

static void sweep(int rows, int block, int t) {
    static struct KeyPoint got[4096];
    static struct KeyPoint want[4096];
    for (int w = 0; w <= 65; w++) {
        struct Mat *m = alloc_mat(rows, w, CV_8UC1);
        fill_random(m, block);
        for (int nonmax = 0; nonmax <= 1; nonmax++) {
            const int32_t nw = ref_fast(want, m, t, nonmax);
            const int32_t ng = fast_detect(got, 4096, m, t, nonmax);
            check_same(got, ng, want, nw);
        }
        free_mat(&m);
    }
}

static void test_fast_width_sweep(void) {
    sweep(9, 1, 20);
    sweep(12, 1, 0);
    sweep(10, 1, 60);
    sweep(11, 2, 10); // equal scores side by side
    sweep(13, 3, 5);
}

static void test_fast_threshold_extremes(void) {
    static struct KeyPoint got[2048];
    static struct KeyPoint want[2048];
    struct Mat            *m = alloc_mat(24, 30, CV_8UC1);
    fill_random(m, 1);
    const int ts[] = {0, 1, 127, 254, 255};
    for (size_t i = 0; i < sizeof ts / sizeof ts[0]; i++) {
        for (int nonmax = 0; nonmax <= 1; nonmax++) {
            const int32_t nw = ref_fast(want, m, ts[i], nonmax);
            check_same(got, fast_detect(got, 2048, m, ts[i], nonmax), want, nw);
        }
    }
    free_mat(&m);
}

// ---- Capacity and misuse ---------------------------------------------------

static void test_fast_capacity(void) {
    static struct KeyPoint all[1024];
    struct KeyPoint        few[4];
    struct Mat            *m = alloc_mat(20, 20, CV_8UC1);
    fill_random(m, 1);
    const int32_t n = fast_detect(all, 1024, m, 10, 1);
    CHECK(n > 4);
    CHECK_EQ_INT(fast_detect(NULL, 0, m, 10, 1), n); // count only
    memset(few, 0xA5, sizeof few);
    CHECK_EQ_INT(fast_detect(few, 3, m, 10, 1), n);
    CHECK_EQ_BYTES(few, all, 3 * sizeof few[0]);
    const uchar *tail = (const uchar *)&few[3];
    for (size_t i = 0; i < sizeof few[3]; i++) {
        CHECK_EQ_INT(tail[i], 0xA5); // nothing written past capacity
    }
    free_mat(&m);
}

static void test_fast_misuse(void) {
    struct KeyPoint kp[1];
    struct Mat     *g = alloc_mat(8, 8, CV_8UC1);
    struct Mat     *c = alloc_mat(8, 8, CV_8UC3);
    EXPECT_ASSERT(fast_detect(kp, 1, c, 10, 1));
    EXPECT_ASSERT(fast_detect(kp, 1, g, -1, 1));
    EXPECT_ASSERT(fast_detect(kp, 1, g, 256, 1));
    EXPECT_ASSERT(fast_detect(kp, -1, g, 10, 1));
    EXPECT_ASSERT(fast_detect(NULL, 1, g, 10, 1));
    EXPECT_ASSERT(fast_detect(kp, 1, NULL, 10, 1));
    free_mat(&g);
    free_mat(&c);
}

static void test_fast_scratch(void) {
    struct KeyPoint kp[16];
    struct Mat     *m = alloc_mat(30, 41, CV_8UC1);
    fill_random(m, 1);
    alloc_counter_start();
    fast_detect(kp, 16, m, 20, 1);
    alloc_counter_stop();
    CHECK(alloc_stats.peak >= 3 * 41); // ring of 3 score rows
    CHECK(alloc_stats.peak <= 3 * 41 + 64);
    CHECK_EQ_INT(alloc_stats.allocs, 1);
    CHECK_EQ_INT(alloc_stats.allocs, alloc_stats.frees);
    alloc_counter_start();
    fast_detect(kp, 16, m, 20, 0);
    alloc_counter_stop();
    CHECK_EQ_INT(alloc_stats.allocs, 0);
    free_mat(&m);
}

// ---- ORB: references -------------------------------------------------------

// Q14 cos and sin of each step, derived here from libm.
static int ref_cs[ORB_ANGLE_STEPS][2];

static void ref_init_cs(void) {
    const double pi = 3.14159265358979323846;
    for (int s = 0; s < ORB_ANGLE_STEPS; s++) {
        const double a = 2.0 * pi * s / ORB_ANGLE_STEPS;
        ref_cs[s][0]   = (int)lround(cos(a) * 16384.0);
        ref_cs[s][1]   = (int)lround(sin(a) * 16384.0);
    }
}

static int64_t ref_harris(const struct Mat *m, int x, int y) {
    int64_t sxx = 0, syy = 0, sxy = 0;
    for (int v = y - 3; v <= y + 3; v++) {
        for (int u = x - 3; u <= x + 3; u++) {
            int64_t gx = 0, gy = 0;
            for (int k = -1; k <= 1; k++) {
                const int w = k == 0 ? 2 : 1;
                gx += w * (ref_px(m, u + 1, v + k) - ref_px(m, u - 1, v + k));
                gy += w * (ref_px(m, u + k, v + 1) - ref_px(m, u + k, v - 1));
            }
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    return 25 * (sxx * syy - sxy * sxy) - (sxx + syy) * (sxx + syy);
}

// Disc half widths built the way cv2's ORB does: rounded circle rows up to
// floor(15 * sqrt(2) / 2 + 1), the rest mirrored so the disc is symmetric.
static int ref_umax[16];

static void ref_init_umax(void) {
    const int vmax = (int)floor(15 * sqrt(2.0) / 2 + 1);
    const int vmin = (int)ceil(15 * sqrt(2.0) / 2);
    for (int v = 0; v <= vmax; v++) {
        ref_umax[v] = (int)lround(sqrt(225.0 - v * v));
    }
    for (int v = 15, v0 = 0; v >= vmin; v--) {
        while (ref_umax[v0] == ref_umax[v0 + 1]) {
            v0++;
        }
        ref_umax[v] = v0;
        v0++;
    }
}

static int ref_angle(const struct Mat *m, int x, int y) {
    int64_t m10 = 0, m01 = 0;
    for (int v = -15; v <= 15; v++) {
        const int umax = ref_umax[v < 0 ? -v : v];
        for (int u = -umax; u <= umax; u++) {
            m10 += (int64_t)u * ref_px(m, x + u, y + v);
            m01 += (int64_t)v * ref_px(m, x + u, y + v);
        }
    }
    int     best = 0;
    int64_t bd   = m10 * ref_cs[0][0] + m01 * ref_cs[0][1];
    for (int s = 1; s < ORB_ANGLE_STEPS; s++) {
        const int64_t d = m10 * ref_cs[s][0] + m01 * ref_cs[s][1];
        if (d > bd) {
            bd   = d;
            best = s;
        }
    }
    return best;
}

static int ref_rot(int a, int b, int c, int s) {
    return (int)lround((a * (double)c - b * (double)s) / 16384.0);
}

static void ref_desc(uint8_t out[ORB_DESC_BYTES], const struct Mat *sm,
                     const struct KeyPoint *kp) {
    const int c = ref_cs[kp->angle][0];
    const int s = ref_cs[kp->angle][1];
    memset(out, 0, ORB_DESC_BYTES);
    for (int k = 0; k < 256; k++) {
        const signed char *p  = improc_orb_pattern[k];
        const int          x1 = ref_rot(p[0], p[1], c, s);
        const int          y1 = ref_rot(p[1], -p[0], c, s); // x*s + y*c
        const int          x2 = ref_rot(p[2], p[3], c, s);
        const int          y2 = ref_rot(p[3], -p[2], c, s);
        if (ref_px(sm, kp->x + x1, kp->y + y1) <
            ref_px(sm, kp->x + x2, kp->y + y2)) {
            out[k / 8] = (uint8_t)(out[k / 8] | (1 << (k % 8)));
        }
    }
}

// Insertion sort of kp[0..k) by key, highest first, ties by raster position.
static void ref_sort(struct KeyPoint *kp, int64_t *key, int32_t k) {
    for (int32_t i = 1; i < k; i++) {
        const struct KeyPoint p  = kp[i];
        const int64_t         kv = key[i];
        int32_t               j  = i;
        while (j > 0 && (key[j - 1] < kv ||
                         (key[j - 1] == kv &&
                          (kp[j - 1].y > p.y ||
                           (kp[j - 1].y == p.y && kp[j - 1].x > p.x))))) {
            kp[j]  = kp[j - 1];
            key[j] = key[j - 1];
            j--;
        }
        kp[j]  = p;
        key[j] = kv;
    }
}

// FAST (non-max) -> border -> best 2N by FAST score -> best N by Harris ->
// angle.
static int32_t ref_orb_detect(struct KeyPoint *out, int32_t max_kps,
                              const struct Mat *m, int t) {
    static struct KeyPoint all[8192];
    static int64_t         key[8192];
    const int32_t          n = ref_fast(all, m, t, 1);
    int32_t                k = 0;
    for (int32_t i = 0; i < n; i++) {
        const int x = all[i].x, y = all[i].y;
        if (x >= 19 && y >= 19 && x < m->cols - 19 && y < m->rows - 19) {
            all[k] = all[i];
            key[k] = all[i].response;
            k++;
        }
    }
    ref_sort(all, key, k);
    if (k > 2 * max_kps) {
        k = 2 * max_kps;
    }
    for (int32_t i = 0; i < k; i++) {
        key[i] = ref_harris(m, all[i].x, all[i].y);
    }
    ref_sort(all, key, k);
    const int32_t kept = k < max_kps ? k : max_kps;
    for (int32_t i = 0; i < kept; i++) {
        out[i]       = all[i];
        out[i].angle = ref_angle(m, out[i].x, out[i].y);
    }
    return kept;
}

static int ref_hamming(const uint8_t *a, const uint8_t *b) {
    int d = 0;
    for (int i = 0; i < 256; i++) {
        d += ((a[i / 8] >> (i % 8)) & 1) != ((b[i / 8] >> (i % 8)) & 1);
    }
    return d;
}

// ---- ORB: tests ------------------------------------------------------------

static uint32_t pat_xorshift(uint32_t s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static void test_orb_pattern_table(void) {
    // Regenerates the table from the rule in features2d.c.
    uint32_t st = 0x9E3779B9u;
    int      n  = 0;
    while (n < 256) {
        int a[4];
        for (int j = 0; j < 4; j++) {
            int sum = 0;
            for (int r = 0; r < 4; r++) {
                st = pat_xorshift(st);
                sum += (int)(st >> 27);
            }
            const int c = sum - 62;
            const int q = (abs(c) + 1) / 3;
            a[j]        = c >= 0 ? q : -q;
        }
        int ok = 1;
        for (int j = 0; j < 4; j++) {
            ok &= a[j] >= -13 && a[j] <= 13;
        }
        if (!ok || (a[0] == a[2] && a[1] == a[3])) {
            continue;
        }
        for (int j = 0; j < 4; j++) {
            if (improc_orb_pattern[n][j] != a[j]) {
                TEST_FAIL_("pair %d[%d] == %d, regenerated %d", n, j,
                           improc_orb_pattern[n][j], a[j]);
                return;
            }
        }
        n++;
    }
}

static void test_orb_harris(void) {
    struct Mat *m = alloc_mat(20, 23, CV_8UC1);
    fill_random(m, 1);
    for (int y = 4; y < 16; y++) {
        for (int x = 4; x < 19; x++) {
            CHECK_EQ_INT(improc_harris_measure(m, x, y), ref_harris(m, x, y));
        }
    }
    // Saturated gradients: 0/255 checkerboard, the largest sums.
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 23; x++) {
            mat_row(m, y)[x] = (uchar)(((x + y) & 1) ? 255 : 0);
        }
    }
    CHECK_EQ_INT(improc_harris_measure(m, 10, 10), ref_harris(m, 10, 10));
    free_mat(&m);
}

static void test_orb_angle_ramps(void) {
    struct Mat *m = alloc_mat(41, 41, CV_8UC1);
    memset(m->data, 77, (size_t)41 * m->step[0]);
    CHECK_EQ_INT(improc_orb_angle(m, 20, 20), 0); // flat: no direction
    for (int s = 0; s < ORB_ANGLE_STEPS; s++) {
        for (int y = 0; y < 41; y++) {
            for (int x = 0; x < 41; x++) {
                const int r = (x - 20) * ref_cs[s][0] + (y - 20) * ref_cs[s][1];
                mat_row(m, y)[x] =
                    (uchar)(128 + (int)lround(3.0 * r / 16384.0));
            }
        }
        CHECK_EQ_INT(improc_orb_angle(m, 20, 20), s);
    }
    fill_random(m, 1);
    for (int y = 15; y < 26; y++) {
        for (int x = 15; x < 26; x++) {
            CHECK_EQ_INT(improc_orb_angle(m, x, y), ref_angle(m, x, y));
        }
    }
    free_mat(&m);
}

static void test_orb_detect_reference(void) {
    static struct KeyPoint got[2048];
    static struct KeyPoint want[2048];
    const int              caps[] = {1, 7, 2048};
    for (int w = 0; w <= 65; w++) {
        struct Mat *m = alloc_mat(w % 2 ? 44 : 52, w, CV_8UC1);
        fill_random(m, 2);
        for (size_t c = 0; c < sizeof caps / sizeof caps[0]; c++) {
            const int32_t nw = ref_orb_detect(want, caps[c], m, 10);
            const int32_t ng = orb_detect(got, caps[c], m, 10);
            CHECK_EQ_INT(ng, nw);
            for (int32_t i = 0; i < ng && i < nw; i++) {
                if (memcmp(&got[i], &want[i], sizeof got[i]) != 0) {
                    TEST_FAIL_("w %d cap %d point %ld: (%ld,%ld,a%ld) want "
                               "(%ld,%ld,a%ld)",
                               w, caps[c], (long)i, (long)got[i].x,
                               (long)got[i].y, (long)got[i].angle,
                               (long)want[i].x, (long)want[i].y,
                               (long)want[i].angle);
                    break;
                }
            }
        }
        free_mat(&m);
    }
}

// A random 8x8 tile repeated over the image: every corner reappears with the
// same FAST score and Harris value, so the selection and its cap depend on
// the raster-order tie rule.
static void test_orb_detect_ties(void) {
    static struct KeyPoint got[2048];
    static struct KeyPoint want[2048];
    struct Mat            *m = alloc_mat(120, 130, CV_8UC1);
    uchar                  tile[8][8];
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            tile[y][x] = (uchar)(xorshift32() >> 24);
        }
    }
    for (int y = 0; y < 120; y++) {
        for (int x = 0; x < 130; x++) {
            mat_row(m, y)[x] = tile[y % 8][x % 8];
        }
    }
    const int caps[] = {1, 5, 20, 100, 2048};
    for (size_t c = 0; c < sizeof caps / sizeof caps[0]; c++) {
        const int32_t nw = ref_orb_detect(want, caps[c], m, 10);
        const int32_t ng = orb_detect(got, caps[c], m, 10);
        CHECK_EQ_INT(ng, nw);
        for (int32_t i = 0; i < ng && i < nw; i++) {
            if (memcmp(&got[i], &want[i], sizeof got[i]) != 0) {
                TEST_FAIL_("cap %d point %ld differs", caps[c], (long)i);
                break;
            }
        }
    }
    free_mat(&m);
}

static void test_orb_compute_reference(void) {
    struct Mat     *m = alloc_mat(60, 80, CV_8UC1);
    struct KeyPoint kp[128];
    uint8_t         got[128][ORB_DESC_BYTES];
    uint8_t         want[ORB_DESC_BYTES];
    fill_random(m, 1);
    int32_t n = 0;
    // Includes the first and last valid rows and columns (19 and size - 20).
    for (int y = 19; y <= 60 - 20; y += 3) {
        for (int x = 19; x <= 80 - 20; x += 5) {
            kp[n].x        = x;
            kp[n].y        = y;
            kp[n].response = 0;
            kp[n].angle    = n % ORB_ANGLE_STEPS;
            n++;
        }
    }
    CHECK(n >= 2 * ORB_ANGLE_STEPS); // every step is used
    orb_compute(got, kp, n, m);
    for (int32_t i = 0; i < n; i++) {
        ref_desc(want, m, &kp[i]);
        if (memcmp(got[i], want, ORB_DESC_BYTES) != 0) {
            TEST_FAIL_("descriptor %ld (angle %ld) differs", (long)i,
                       (long)kp[i].angle);
            break;
        }
    }
    orb_compute(NULL, NULL, 0, m); // nothing to do
    free_mat(&m);
}

// Exact 90-degree rotation: B(rows-1-y, x) = A(x, y). FAST, Harris, the
// angle steps and the blur are all symmetric under it, so every keypoint and
// descriptor must reappear, with the angle 8 steps further.
static void test_orb_rot90_invariance(void) {
    static struct KeyPoint ka[1024];
    static struct KeyPoint kb[1024];
    static uint8_t         da[1024][ORB_DESC_BYTES];
    static uint8_t         db[1024][ORB_DESC_BYTES];
    struct Mat            *a  = alloc_mat(60, 83, CV_8UC1);
    struct Mat            *b  = alloc_mat(83, 60, CV_8UC1);
    struct Mat            *sa = alloc_mat(60, 83, CV_8UC1);
    struct Mat            *sb = alloc_mat(83, 60, CV_8UC1);
    fill_random(a, 3);
    for (int y = 0; y < 60; y++) {
        for (int x = 0; x < 83; x++) {
            mat_row(b, x)[60 - 1 - y] = mat_row(a, y)[x];
        }
    }
    gaussian_blur(sa, a, 5);
    gaussian_blur(sb, b, 5);
    const int32_t na = orb_detect(ka, 1024, a, 10);
    const int32_t nb = orb_detect(kb, 1024, b, 10);
    CHECK(na > 10);
    CHECK(na < 1024);
    CHECK_EQ_INT(nb, na);
    orb_compute(da, ka, na, sa);
    orb_compute(db, kb, nb, sb);
    int matched = 0;
    for (int32_t i = 0; i < na; i++) {
        for (int32_t j = 0; j < nb; j++) {
            if (kb[j].x == 60 - 1 - ka[i].y && kb[j].y == ka[i].x) {
                CHECK_EQ_INT(kb[j].angle, (ka[i].angle + 8) % ORB_ANGLE_STEPS);
                CHECK_EQ_BYTES(db[j], da[i], ORB_DESC_BYTES);
                matched++;
                break;
            }
        }
    }
    CHECK_EQ_INT(matched, na);
    free_mat(&a);
    free_mat(&b);
    free_mat(&sa);
    free_mat(&sb);
}

static void test_hamming(void) {
    uint8_t a[ORB_DESC_BYTES], b[ORB_DESC_BYTES];
    memset(a, 0, sizeof a);
    memset(b, 0xFF, sizeof b);
    CHECK_EQ_INT(hamming_distance(a, b), 256);
    CHECK_EQ_INT(hamming_distance(a, a), 0);
    b[0] = 0;
    memset(b + 1, 0, sizeof b - 1);
    b[31] = 0x80;
    CHECK_EQ_INT(hamming_distance(a, b), 1);
    for (int r = 0; r < 200; r++) {
        for (int i = 0; i < ORB_DESC_BYTES; i++) {
            a[i] = (uint8_t)(xorshift32() >> 24);
            b[i] = (uint8_t)(xorshift32() >> 24);
        }
        CHECK_EQ_INT(hamming_distance(a, b), ref_hamming(a, b));
    }
}

static void fill_desc(uint8_t (*d)[ORB_DESC_BYTES], int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < ORB_DESC_BYTES; k++) {
            d[i][k] = (uint8_t)(xorshift32() >> 24);
        }
    }
}

static void test_match_hand(void) {
    uint8_t q[3][ORB_DESC_BYTES], t[3][ORB_DESC_BYTES];
    memset(q, 0, sizeof q);
    memset(t, 0, sizeof t);
    q[1][0] = 0x0F; // q0 = 0, q1 = 4 bits, q2 = 0
    t[0][0] = 0x01; // t0 = 1 bit, t1 = 0x0F, t2 = 0x0F (tie with t1)
    t[1][0] = 0x0F;
    t[2][0] = 0x0F;
    struct DMatch mt[3];
    CHECK_EQ_INT(match_descriptors(mt, q, 3, t, 3, 0), 3);
    CHECK_EQ_INT(mt[0].train, 0);
    CHECK_EQ_INT(mt[0].distance, 1);
    CHECK_EQ_INT(mt[1].train, 1); // tie t1/t2: lowest index
    CHECK_EQ_INT(mt[1].distance, 0);
    CHECK_EQ_INT(mt[2].query, 2);
    CHECK_EQ_INT(mt[2].train, 0);
    // Cross check: t0's best query is q0 (tie q0/q2: lowest), t1's is q1.
    CHECK_EQ_INT(match_descriptors(mt, q, 3, t, 3, 1), 2);
    CHECK_EQ_INT(mt[0].query, 0);
    CHECK_EQ_INT(mt[0].train, 0);
    CHECK_EQ_INT(mt[1].query, 1);
    CHECK_EQ_INT(mt[1].train, 1);
    CHECK_EQ_INT(match_descriptors(mt, q, 0, t, 3, 1), 0);
    CHECK_EQ_INT(match_descriptors(mt, q, 3, NULL, 0, 1), 0);
}

static void test_match_reference(void) {
    static uint8_t       q[70][ORB_DESC_BYTES], t[50][ORB_DESC_BYTES];
    static struct DMatch mt[70];
    for (int nq = 1; nq <= 70; nq += 23) {
        for (int nt = 1; nt <= 50; nt += 7) {
            fill_desc(q, nq);
            fill_desc(t, nt);
            for (int cc = 0; cc <= 1; cc++) {
                int32_t want = 0;
                for (int i = 0; i < nq; i++) {
                    int best = 0;
                    for (int j = 1; j < nt; j++) {
                        if (ref_hamming(q[i], t[j]) <
                            ref_hamming(q[i], t[best])) {
                            best = j;
                        }
                    }
                    int back = 0;
                    for (int k = 1; k < nq; k++) {
                        if (ref_hamming(q[k], t[best]) <
                            ref_hamming(q[back], t[best])) {
                            back = k;
                        }
                    }
                    if (cc && back != i) {
                        continue;
                    }
                    want++;
                }
                const int32_t got = match_descriptors(mt, q, nq, t, nt, cc);
                CHECK_EQ_INT(got, want);
                int32_t k = 0;
                for (int i = 0; i < nq && k < got; i++) {
                    int best = 0;
                    for (int j = 1; j < nt; j++) {
                        if (ref_hamming(q[i], t[j]) <
                            ref_hamming(q[i], t[best])) {
                            best = j;
                        }
                    }
                    int back = 0;
                    for (int r = 1; r < nq; r++) {
                        if (ref_hamming(q[r], t[best]) <
                            ref_hamming(q[back], t[best])) {
                            back = r;
                        }
                    }
                    if (cc && back != i) {
                        continue;
                    }
                    CHECK_EQ_INT(mt[k].query, i);
                    CHECK_EQ_INT(mt[k].train, best);
                    CHECK_EQ_INT(mt[k].distance, ref_hamming(q[i], t[best]));
                    k++;
                }
            }
        }
    }
}

static void test_orb_misuse(void) {
    struct KeyPoint kp[1] = {{19, 19, 0, 0}};
    uint8_t         d[1][ORB_DESC_BYTES];
    struct DMatch   mt[1];
    struct Mat     *g = alloc_mat(40, 40, CV_8UC1);
    struct Mat     *c = alloc_mat(40, 40, CV_8UC3);
    EXPECT_ASSERT(orb_detect(kp, 1, c, 10));
    EXPECT_ASSERT(orb_detect(kp, 1, g, 256));
    EXPECT_ASSERT(orb_detect(kp, -1, g, 10));
    EXPECT_ASSERT(orb_detect(NULL, 1, g, 10));
    EXPECT_ASSERT(orb_compute(d, kp, 1, c));
    kp[0].angle = ORB_ANGLE_STEPS;
    EXPECT_ASSERT(orb_compute(d, kp, 1, g));
    kp[0].angle = 0;
    kp[0].x     = 18; // inside the border
    EXPECT_ASSERT(orb_compute(d, kp, 1, g));
    kp[0].x = 40 - 19;
    EXPECT_ASSERT(orb_compute(d, kp, 1, g));
    kp[0].x = 19;
    orb_compute(d, kp, 1, g); // the corner of the valid area is fine
    EXPECT_ASSERT(orb_compute(d, kp, -1, g));
    EXPECT_ASSERT(match_descriptors(mt, d, -1, d, 1, 0));
    EXPECT_ASSERT(match_descriptors(NULL, d, 1, d, 1, 0));
    EXPECT_ASSERT(match_descriptors(mt, d, 1, NULL, 1, 0));
    free_mat(&g);
    free_mat(&c);
}

static void test_orb_scratch(void) {
    static struct KeyPoint kp[64];
    static uint8_t         d[40][ORB_DESC_BYTES];
    static struct DMatch   mt[40];
    struct Mat            *m = alloc_mat(50, 61, CV_8UC1);
    fill_random(m, 2);
    const int32_t n = fast_detect(NULL, 0, m, 10, 1);
    alloc_counter_start();
    orb_detect(kp, 64, m, 10);
    alloc_counter_stop();
    const size_t limit =
        (size_t)n * (sizeof(struct KeyPoint) + 16) + 3 * 61 + 3 * 64;
    CHECK(alloc_stats.peak >= (size_t)n * sizeof(struct KeyPoint));
    CHECK(alloc_stats.peak <= limit);
    CHECK_EQ_INT(alloc_stats.allocs, alloc_stats.frees);
    fill_desc(d, 40);
    alloc_counter_start();
    match_descriptors(mt, d, 40, d, 30, 1);
    alloc_counter_stop();
    CHECK_EQ_INT(alloc_stats.allocs, 1);
    CHECK(alloc_stats.peak >= 30 * 4 && alloc_stats.peak <= 30 * 4 + 64);
    alloc_counter_start();
    match_descriptors(mt, d, 40, d, 30, 0);
    orb_compute(d, kp, 0, m);
    alloc_counter_stop();
    CHECK_EQ_INT(alloc_stats.allocs, 0);
    free_mat(&m);
}

int main(void) {
    ref_init_cs();
    ref_init_umax();
    RUN_TEST(test_fast_bright_arc);
    RUN_TEST(test_fast_dark_arc_wraps);
    RUN_TEST(test_fast_arc_of_8_is_not_a_corner);
    RUN_TEST(test_fast_score_uses_weakest_pixel);
    RUN_TEST(test_fast_score_one_corner);
    RUN_TEST(test_fast_small_images);
    RUN_TEST(test_fast_width_sweep);
    RUN_TEST(test_fast_threshold_extremes);
    RUN_TEST(test_fast_capacity);
    RUN_TEST(test_fast_misuse);
    RUN_TEST(test_fast_scratch);
    RUN_TEST(test_orb_pattern_table);
    RUN_TEST(test_orb_harris);
    RUN_TEST(test_orb_angle_ramps);
    RUN_TEST(test_orb_detect_reference);
    RUN_TEST(test_orb_detect_ties);
    RUN_TEST(test_orb_compute_reference);
    RUN_TEST(test_orb_rot90_invariance);
    RUN_TEST(test_hamming);
    RUN_TEST(test_match_hand);
    RUN_TEST(test_match_reference);
    RUN_TEST(test_orb_misuse);
    RUN_TEST(test_orb_scratch);
    return test_finish();
}
