#include <stdint.h>
#include <string.h>

#include "improc.h"
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

int main(void) {
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
    return test_finish();
}
