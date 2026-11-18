#include <stdint.h>
#include <string.h>

#include "improc.h"
#include "test_alloc_util.h"
#include "test_expect_assert.h"
#include "test_mat_util.h"

// ---- Helpers and naive references -------------------------------------------

static uint32_t rng_state = 0x12345678u;

static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

// Uniform in lo..hi (hi - lo < 2^32).
static int64_t rnd(int64_t lo, int64_t hi) {
    return lo + (int64_t)(xorshift32() % (uint32_t)(hi - lo + 1));
}

static const uchar WHITE[4] = {255, 255, 255, 255};

// Image of the given type whose bytes are a pattern, so untouched pixels and
// untouched channels can be told apart from drawn ones.
static struct Mat *canvas(int rows, int cols, int type) {
    struct Mat *m = alloc_mat(rows, cols, type);
    if (rows > 0 && cols > 0) {
        fill_pattern(m, 3);
    }
    return m;
}

static struct Mat *zeros(int rows, int cols) {
    struct Mat *m = alloc_mat(rows, cols, CV_8UC1);
    if (rows > 0 && cols > 0) {
        memset(m->data, 0, (size_t)rows * m->step[0]);
    }
    return m;
}

static int px(const struct Mat *m, int x, int y) { return mat_row(m, y)[x]; }

// Compares two CV_8UC1 images and reports the first differing pixel.
static int same_image(const struct Mat *a, const struct Mat *b) {
    for (int y = 0; y < a->rows; y++) {
        for (int x = 0; x < a->cols; x++) {
            if (px(a, x, y) != px(b, x, y)) {
                printf("    first difference at (%d, %d): %d vs %d\n", x, y,
                       px(a, x, y), px(b, x, y));
                return 0;
            }
        }
    }
    return 1;
}

// Integer square root by binary search (unlike the library's bitwise one).
static int64_t ref_isqrt(int64_t n) {
    int64_t lo = 0, hi = 3037000499LL; // floor(sqrt(2^63 - 1))
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (mid <= n / mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// cv2.clipLine with plain int64 arithmetic. Only valid while the products fit
// int64_t, so the sweeps keep |coordinates| <= 2^28.
static int ref_clip(int64_t w, int64_t h, int64_t *x1, int64_t *y1, int64_t *x2,
                    int64_t *y2) {
    const int64_t r = w - 1, b = h - 1;
#define CODE(x, y)                                                             \
    (((x) < 0) | ((x) > r) << 1 | ((y) < 0) << 2 | ((y) > b) << 3)
    int c1 = CODE(*x1, *y1), c2 = CODE(*x2, *y2);
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
        if (c1 & 12) {
            const int64_t a = c1 < 8 ? 0 : b;
            *x1 += (a - *y1) * (*x2 - *x1) / (*y2 - *y1);
            *y1 = a;
            c1  = CODE(*x1, 0);
        }
        if (c2 & 12) {
            const int64_t a = c2 < 8 ? 0 : b;
            *x2 += (a - *y2) * (*x2 - *x1) / (*y2 - *y1);
            *y2 = a;
            c2  = CODE(*x2, 0);
        }
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if (c1) {
                const int64_t a = c1 == 1 ? 0 : r;
                *y1 += (a - *x1) * (*y2 - *y1) / (*x2 - *x1);
                *x1 = a;
                c1  = 0;
            }
            if (c2) {
                const int64_t a = c2 == 1 ? 0 : r;
                *y2 += (a - *x2) * (*y2 - *y1) / (*x2 - *x1);
                *x2 = a;
                c2  = 0;
            }
        }
    }
#undef CODE
    return (c1 | c2) == 0;
}

// Points of the cv2 LINE_8 line clipped to [0, w) x [0, h): pixel i of the
// major axis has minor offset round(i * dy / dx) with ties rounded down,
// written in closed form instead of an error term. Returns the count.
static int ref_line_points(int64_t w, int64_t h, int64_t x1, int64_t y1,
                           int64_t x2, int64_t y2, int64_t (*pts)[2], int cap) {
    if (w <= 0 || h <= 0 || !ref_clip(w, h, &x1, &y1, &x2, &y2)) {
        return 0;
    }
    if (x2 < x1) {
        int64_t t = x1;
        x1        = x2;
        x2        = t;
        t         = y1;
        y1        = y2;
        y2        = t;
    }
    const int64_t dx = x2 - x1, ady = y2 > y1 ? y2 - y1 : y1 - y2;
    const int64_t sy    = y2 < y1 ? -1 : 1;
    const int     vert  = ady > dx;
    const int64_t major = vert ? ady : dx, minor = vert ? dx : ady;
    int           n = 0;
    for (int64_t i = 0; i <= major; i++) {
        const int64_t off =
            major == 0 ? 0 : (2 * minor * i + major - 1) / (2 * major);
        CHECK(n < cap);
        if (n >= cap) {
            break;
        }
        pts[n][0] = vert ? x1 + off : x1 + i;
        pts[n][1] = vert ? y1 + sy * i : y1 + sy * off;
        n++;
    }
    return n;
}

#define MAX_PTS 4096
static int64_t ref_pts[MAX_PTS][2];

static void ref_line(struct Mat *m, int64_t x1, int64_t y1, int64_t x2,
                     int64_t y2) {
    const int n =
        ref_line_points(m->cols, m->rows, x1, y1, x2, y2, ref_pts, MAX_PTS);
    for (int i = 0; i < n; i++) {
        mat_row(m, (int)ref_pts[i][1])[ref_pts[i][0]] = 255;
    }
}

// Thick line: every pixel within the brush of some point of the thin line
// walked in the image grown by t / 2 (draw.h). Pixel by pixel, no stamping.
static void ref_thick_line(struct Mat *m, int64_t x1, int64_t y1, int64_t x2,
                           int64_t y2, int t) {
    const int64_t g = t / 2, e = (t + 1) & 1;
    const int     n = ref_line_points(m->cols + 2 * g, m->rows + 2 * g, x1 + g,
                                      y1 + g, x2 + g, y2 + g, ref_pts, MAX_PTS);
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->cols; x++) {
            for (int k = 0; k < n; k++) {
                const int64_t i = x - (ref_pts[k][0] - g);
                const int64_t j = y - (ref_pts[k][1] - g);
                if ((2 * i - e) * (2 * i - e) + (2 * j - e) * (2 * j - e) <=
                    (int64_t)t * t) {
                    mat_row(m, y)[x] = 255;
                    break;
                }
            }
        }
    }
}

// Circle membership of the offset (a, b) from the centre, from the definition
// in draw.h. a, b and r are below 2^31 in magnitude.
static int ref_on_circle(int64_t a, int64_t b, int64_t r, int thickness) {
    const int64_t d2 = a * a + b * b; // < 2^63
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
    if (thickness < 0) {
        return a <= r && b <= r && d2 <= r * r;
    }
    if (thickness == 1) {
        if (a > r || b > r) {
            return 0;
        }
        return (2 * b * b <= r * r && a == ref_isqrt(r * r - b * b)) ||
               (2 * a * a <= r * r && b == ref_isqrt(r * r - a * a));
    }
    const int64_t ro = r + thickness / 2, ri = ro - thickness;
    if (a > ro || b > ro || d2 > ro * ro + ro) {
        return 0;
    }
    return !(ri >= 0 && d2 <= ri * ri + ri);
}

static void ref_circle(struct Mat *m, int64_t cx, int64_t cy, int64_t r,
                       int thickness) {
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->cols; x++) {
            if (ref_on_circle(x - cx, y - cy, r, thickness)) {
                mat_row(m, y)[x] = 255;
            }
        }
    }
}

static int count_nonzero(const struct Mat *m) {
    int n = 0;
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->cols; x++) {
            n += px(m, x, y) != 0;
        }
    }
    return n;
}

// Parses rows of '#' and '.' into the expected CV_8UC1 image (255 / 0).
static void check_art(const struct Mat *m, const char *const *art) {
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->cols; x++) {
            const int want = art[y][x] == '#' ? 255 : 0;
            if (px(m, x, y) != want) {
                TEST_FAIL_("pixel (%d, %d) == %d, expected %d", x, y,
                           px(m, x, y), want);
                return;
            }
        }
    }
}

// ---- Points -----------------------------------------------------------------

static void test_point_channels(void) {
    const int   types[3] = {CV_8UC1, CV_8UC3, CV_8UC4};
    const uchar color[4] = {11, 22, 33, 44};
    for (int t = 0; t < 3; t++) {
        struct Mat  *m  = canvas(3, 4, types[t]);
        struct Mat  *e  = canvas(3, 4, types[t]);
        const size_t cn = mat_channel_num(m);
        draw_point(m, 2, 1, color);
        memcpy(mat_row(e, 1) + 2 * cn, color, cn);
        // Outside: nothing changes.
        draw_point(m, -1, 0, color);
        draw_point(m, 4, 0, color);
        draw_point(m, 0, 3, color);
        draw_point(m, INT32_MIN, INT32_MAX, color);
        CHECK_MAT_BYTES(m, e->data);
        free_mat(&m);
        free_mat(&e);
    }
}

// ---- Lines ------------------------------------------------------------------

static void test_line_hand(void) {
    // dx 6, dy 2: minor offsets round(i / 3), ties down: 0 0 1 1 1 2 2.
    static const char *const art[] = {
        "##.....",
        "..###..",
        ".....##",
        ".......",
    };
    struct Mat *m = zeros(4, 7);
    draw_line(m, 0, 0, 6, 2, 1, WHITE);
    check_art(m, art);
    // The other direction draws the same pixels (walked from the left end).
    struct Mat *r = zeros(4, 7);
    draw_line(r, 6, 2, 0, 0, 1, WHITE);
    CHECK(same_image(m, r));

    // Steep: dy 4, dx 1 -> x offsets round(i / 4): 0 0 0 1 1 (ties down).
    static const char *const steep[] = {
        "#..", "#..", "#..", ".#.", ".#.",
    };
    struct Mat *s = zeros(5, 3);
    draw_line(s, 0, 0, 1, 4, 1, WHITE);
    check_art(s, steep);

    // Equal end points: one pixel.
    struct Mat *p = zeros(3, 3);
    draw_line(p, 1, 2, 1, 2, 1, WHITE);
    CHECK_EQ_INT(count_nonzero(p), 1);
    CHECK_EQ_INT(px(p, 1, 2), 255);
    free_mat(&m);
    free_mat(&r);
    free_mat(&s);
    free_mat(&p);
}

static void test_line_channels(void) {
    const uchar color[4] = {10, 20, 30, 40};
    const int   types[2] = {CV_8UC3, CV_8UC4};
    for (int t = 0; t < 2; t++) {
        struct Mat  *m  = canvas(4, 5, types[t]);
        struct Mat  *e  = canvas(4, 5, types[t]);
        const size_t cn = mat_channel_num(m);
        draw_line(m, 0, 3, 4, 3, 1, color); // bottom row
        for (int x = 0; x < 5; x++) {
            memcpy(mat_row(e, 3) + (size_t)x * cn, color, cn);
        }
        CHECK_MAT_BYTES(m, e->data);
        free_mat(&m);
        free_mat(&e);
    }
}

// Widths 0..65 (and a few heights) against the reference, with end points
// inside, near and far outside the image.
static void test_line_sweep(void) {
    const int heights[4] = {0, 1, 7, 33};
    for (int hi = 0; hi < 4; hi++) {
        for (int w = 0; w <= 65; w++) {
            const int h = heights[hi];
            for (int trial = 0; trial < 12; trial++) {
                const int64_t spread = trial < 4 ? 0 : trial < 8 ? 20 : 3000;
                const int64_t x1     = rnd(-spread, w + spread);
                const int64_t y1     = rnd(-spread, h + spread);
                int64_t       x2     = rnd(-spread, w + spread);
                int64_t       y2     = rnd(-spread, h + spread);
                if (trial == 3) {
                    x2 = x1;
                    y2 = y1;
                }
                struct Mat *a = zeros(h, w);
                struct Mat *b = zeros(h, w);
                draw_line(a, (int32_t)x1, (int32_t)y1, (int32_t)x2, (int32_t)y2,
                          1, WHITE);
                ref_line(b, x1, y1, x2, y2);
                if (!same_image(a, b)) {
                    TEST_FAIL_("w %d h %d line (%lld, %lld)-(%lld, %lld)", w, h,
                               (long long)x1, (long long)y1, (long long)x2,
                               (long long)y2);
                }
                free_mat(&a);
                free_mat(&b);
            }
        }
    }
}

static void test_line_clipping(void) {
    struct Mat *m = zeros(10, 12);
    struct Mat *r = zeros(10, 12);

    // Fully outside: left of, above, and crossing only the corner region.
    draw_line(m, -5, -3, -1, 20, 1, WHITE);
    draw_line(m, 0, -1, 11, -1, 1, WHITE);
    draw_line(m, -10, 3, 3, -10, 1, WHITE);
    CHECK_EQ_INT(count_nonzero(m), 0);

    // Crossing the border: same pixels as the reference (clip, then walk).
    draw_line(m, -7, 2, 20, 9, 1, WHITE);
    draw_line(m, 5, -30, 8, 40, 1, WHITE);
    ref_line(r, -7, 2, 20, 9);
    ref_line(r, 5, -30, 8, 40);
    CHECK(same_image(m, r));
    free_mat(&m);
    free_mat(&r);
}

// int32_t extremes: exact expected pixels (the reference cannot hold these
// products) and no long loop.
static void test_line_huge(void) {
    struct Mat *m = zeros(30, 40);
    draw_line(m, INT32_MIN, 5, INT32_MAX, 5, 1, WHITE);
    for (int x = 0; x < 40; x++) {
        CHECK_EQ_INT(px(m, x, 5), 255);
    }
    CHECK_EQ_INT(count_nonzero(m), 40);

    struct Mat *v = zeros(30, 40);
    draw_line(v, 3, INT32_MAX, 3, INT32_MIN, 1, WHITE);
    for (int y = 0; y < 30; y++) {
        CHECK_EQ_INT(px(v, 3, y), 255);
    }
    CHECK_EQ_INT(count_nonzero(v), 30);

    // The main diagonal, clipped from both extremes.
    struct Mat *d = zeros(30, 40);
    draw_line(d, INT32_MIN, INT32_MIN, INT32_MAX, INT32_MAX, 1, WHITE);
    for (int i = 0; i < 30; i++) {
        CHECK_EQ_INT(px(d, i, i), 255);
    }
    CHECK_EQ_INT(count_nonzero(d), 30);

    // The anti-diagonal through (-1, 0) and (0, -1) misses the image.
    struct Mat *o = zeros(30, 40);
    draw_line(o, INT32_MIN, INT32_MAX, INT32_MAX, INT32_MIN, 1, WHITE);
    CHECK_EQ_INT(count_nonzero(o), 0);

    // From inside to 2^28 away: against the reference (products < 2^63).
    for (int trial = 0; trial < 200; trial++) {
        struct Mat   *a  = zeros(30, 40);
        struct Mat   *b  = zeros(30, 40);
        const int64_t x1 = rnd(0, 39), y1 = rnd(0, 29);
        const int64_t x2 = rnd(-(1 << 28), 1 << 28);
        const int64_t y2 = rnd(-(1 << 28), 1 << 28);
        draw_line(a, (int32_t)x1, (int32_t)y1, (int32_t)x2, (int32_t)y2, 1,
                  WHITE);
        ref_line(b, x1, y1, x2, y2);
        CHECK(same_image(a, b));
        // Thick, from the far end: against the pixel-by-pixel reference.
        memset(a->data, 0, (size_t)30 * a->step[0]);
        memset(b->data, 0, (size_t)30 * b->step[0]);
        draw_line(a, (int32_t)x2, (int32_t)y2, (int32_t)x1, (int32_t)y1, 9,
                  WHITE);
        ref_thick_line(b, x2, y2, x1, y1, 9);
        CHECK(same_image(a, b));
        free_mat(&a);
        free_mat(&b);
    }
    free_mat(&m);
    free_mat(&v);
    free_mat(&d);
    free_mat(&o);
}

static void test_thick_line_hand(void) {
    // t 2: 2x2 brush at offsets 0..1 (half a pixel right of and below).
    static const char *const t2[] = {
        "......", "......", ".#####", ".#####", "......", "......",
    };
    struct Mat *m = zeros(6, 6);
    draw_line(m, 1, 2, 4, 2, 2, WHITE);
    check_art(m, t2);
    // t 3: 3x3 brush (i^2 + j^2 <= 2), so rows 1..3 and columns 0..5.
    static const char *const t3[] = {
        "......", "######", "######", "######", "......", "......",
    };
    struct Mat *n = zeros(6, 6);
    draw_line(n, 1, 2, 4, 2, 3, WHITE);
    check_art(n, t3);
    // t 5 dot: 5x5 without its four corners.
    static const char *const t5[] = {
        ".###...", "#####..", "#####..", "#####..", ".###...",
    };
    struct Mat *d = zeros(5, 7);
    draw_line(d, 2, 2, 2, 2, 5, WHITE);
    check_art(d, t5);
    free_mat(&m);
    free_mat(&n);
    free_mat(&d);
}

// The largest thickness on int32_t-extreme lines: a horizontal band of rows
// y - 127 .. y + 127 (odd t, brush reach 127 at its centre column), and the
// whole image for a line through it. These must also finish quickly.
static void test_thick_line_max(void) {
    struct Mat *m = zeros(300, 400);
    draw_line(m, INT32_MIN, 150, INT32_MAX, 150, DRAW_MAX_THICKNESS, WHITE);
    for (int y = 0; y < 300; y++) {
        const int want = (y >= 150 - 127 && y <= 150 + 127) ? 255 : 0;
        CHECK_EQ_INT(px(m, 200, y), want);
    }
    struct Mat *s = zeros(40, 50);
    draw_line(s, INT32_MIN, INT32_MIN, INT32_MAX, INT32_MAX, DRAW_MAX_THICKNESS,
              WHITE);
    CHECK_EQ_INT(count_nonzero(s), 40 * 50);
    struct Mat *r = zeros(40, 50);
    struct Mat *q = zeros(40, 50);
    draw_line(r, -300, 10, 350, 33, 64, WHITE);
    ref_thick_line(q, -300, 10, 350, 33, 64);
    CHECK(same_image(r, q));
    free_mat(&m);
    free_mat(&s);
    free_mat(&r);
    free_mat(&q);
}

static void test_thick_line_sweep(void) {
    for (int w = 0; w <= 65; w += 5) {
        for (int t = 2; t <= 7; t++) {
            for (int trial = 0; trial < 3; trial++) {
                const int     h  = 17;
                const int64_t x1 = rnd(-12, w + 12), y1 = rnd(-12, h + 12);
                const int64_t x2 = rnd(-12, w + 12), y2 = rnd(-12, h + 12);
                struct Mat   *a = zeros(h, w);
                struct Mat   *b = zeros(h, w);
                draw_line(a, (int32_t)x1, (int32_t)y1, (int32_t)x2, (int32_t)y2,
                          t, WHITE);
                ref_thick_line(b, x1, y1, x2, y2, t);
                if (!same_image(a, b)) {
                    TEST_FAIL_("w %d t %d line (%lld, %lld)-(%lld, %lld)", w, t,
                               (long long)x1, (long long)y1, (long long)x2,
                               (long long)y2);
                }
                free_mat(&a);
                free_mat(&b);
            }
        }
    }
}

// ---- Circles ----------------------------------------------------------------

static void test_circle_hand(void) {
    static const char *const r0[] = {"...", ".#.", "..."};
    static const char *const r1[] = {".....", "..#..", ".#.#.", "..#..",
                                     "....."};
    static const char *const r2[] = {
        "..#..", ".#.#.", "#...#", ".#.#.", "..#..",
    };
    static const char *const f2[] = {
        "..#..", ".###.", "#####", ".###.", "..#..",
    };
    // Ring t 2 around r 2: ro 3, ri 1 -> 1 + 1 < d^2 <= 9 + 3.
    static const char *const ring[] = {
        "..###..", ".#####.", "##...##", "##...##",
        "##...##", ".#####.", "..###..",
    };
    struct Mat *a = zeros(3, 3);
    draw_circle(a, 1, 1, 0, 1, WHITE);
    check_art(a, r0);
    struct Mat *a2 = zeros(3, 3);
    draw_circle(a2, 1, 1, 0, -1, WHITE);
    check_art(a2, r0);
    struct Mat *b = zeros(5, 5);
    draw_circle(b, 2, 2, 1, 1, WHITE);
    check_art(b, r1);
    struct Mat *c = zeros(5, 5);
    draw_circle(c, 2, 2, 2, 1, WHITE);
    check_art(c, r2);
    struct Mat *d = zeros(5, 5);
    draw_circle(d, 2, 2, 2, -1, WHITE);
    check_art(d, f2);
    struct Mat *e = zeros(7, 7);
    draw_circle(e, 3, 3, 2, 2, WHITE);
    check_art(e, ring);
    free_mat(&a);
    free_mat(&a2);
    free_mat(&b);
    free_mat(&c);
    free_mat(&d);
    free_mat(&e);
}

static void test_circle_sweep(void) {
    const int thick[6] = {1, -1, 2, 3, 4, 7};
    for (int w = 0; w <= 65; w++) {
        for (int ti = 0; ti < 6; ti++) {
            for (int trial = 0; trial < 3; trial++) {
                const int     h  = 1 + (w * 7 + trial) % 23;
                const int64_t r  = trial == 0 ? rnd(0, 3) : rnd(0, 40);
                const int64_t cx = rnd(-45, w + 45), cy = rnd(-45, h + 45);
                struct Mat   *a = zeros(h, w);
                struct Mat   *b = zeros(h, w);
                draw_circle(a, (int32_t)cx, (int32_t)cy, (int32_t)r, thick[ti],
                            WHITE);
                ref_circle(b, cx, cy, r, thick[ti]);
                if (!same_image(a, b)) {
                    TEST_FAIL_("w %d h %d circle (%lld, %lld) r %lld t %d", w,
                               h, (long long)cx, (long long)cy, (long long)r,
                               thick[ti]);
                }
                free_mat(&a);
                free_mat(&b);
            }
        }
    }
}

// Clipping is per pixel: drawing on a window of a larger canvas equals the
// matching crop of the canvas.
static void test_circle_window(void) {
    struct Mat *big      = zeros(60, 70);
    struct Mat *win      = zeros(13, 17);
    struct Mat *cut      = zeros(13, 17);
    const int   thick[4] = {1, -1, 3, 6};
    for (int ti = 0; ti < 4; ti++) {
        for (int trial = 0; trial < 20; trial++) {
            const int cx = (int)rnd(0, 69), cy = (int)rnd(0, 59);
            const int r = (int)rnd(0, 30), ox = (int)rnd(0, 53),
                      oy = (int)rnd(0, 47);
            memset(big->data, 0, (size_t)60 * big->step[0]);
            memset(win->data, 0, (size_t)13 * win->step[0]);
            draw_circle(big, cx, cy, r, thick[ti], WHITE);
            draw_circle(win, cx - ox, cy - oy, r, thick[ti], WHITE);
            crop(cut, big, ox, oy);
            CHECK(same_image(win, cut));
        }
    }
    free_mat(&big);
    free_mat(&win);
    free_mat(&cut);
}

// Huge centre and radius: the circle still passes through the image where
// expected, and the run time does not depend on the radius.
static void test_circle_huge(void) {
    const int32_t big = 1 << 30;
    // Centre far left, radius reaching column 5 at row 10.
    struct Mat *a = zeros(20, 30);
    struct Mat *b = zeros(20, 30);
    draw_circle(a, 5 - big, 10, big, 1, WHITE);
    ref_circle(b, 5 - (int64_t)big, 10, big, 1);
    CHECK(same_image(a, b));
    CHECK_EQ_INT(px(a, 5, 10), 255);
    CHECK_EQ_INT(count_nonzero(a), 20); // a vertical run near the tangent
    // Same for the filled circle and a thick ring.
    memset(a->data, 0, (size_t)20 * a->step[0]);
    memset(b->data, 0, (size_t)20 * b->step[0]);
    draw_circle(a, 5 - big, 10, big, -1, WHITE);
    ref_circle(b, 5 - (int64_t)big, 10, big, -1);
    CHECK(same_image(a, b));
    memset(a->data, 0, (size_t)20 * a->step[0]);
    memset(b->data, 0, (size_t)20 * b->step[0]);
    draw_circle(a, 5 - big, 10, big, 9, WHITE);
    ref_circle(b, 5 - (int64_t)big, 10, big, 9);
    CHECK(same_image(a, b));

    // The image inside a huge outline: nothing drawn. Filled: everything.
    struct Mat *c = zeros(20, 30);
    draw_circle(c, 15, 10, INT32_MAX, 1, WHITE);
    CHECK_EQ_INT(count_nonzero(c), 0);
    draw_circle(c, 15, 10, INT32_MAX, DRAW_MAX_THICKNESS, WHITE);
    CHECK_EQ_INT(count_nonzero(c), 0);
    draw_circle(c, 15, 10, INT32_MAX, -1, WHITE);
    CHECK_EQ_INT(count_nonzero(c), 600);

    // Centre at the int32_t extremes, far from the image.
    struct Mat *d = zeros(20, 30);
    draw_circle(d, INT32_MIN, INT32_MAX, 1000, 1, WHITE);
    draw_circle(d, INT32_MAX, INT32_MIN, INT32_MAX, -1, WHITE);
    draw_circle(d, INT32_MIN, INT32_MIN, INT32_MAX, 5, WHITE);
    CHECK_EQ_INT(count_nonzero(d), 0);
    free_mat(&a);
    free_mat(&b);
    free_mat(&c);
    free_mat(&d);
}

static void test_circle_channels(void) {
    const uchar color[4] = {7, 8, 9, 10};
    struct Mat *m        = canvas(5, 5, CV_8UC4);
    struct Mat *e        = canvas(5, 5, CV_8UC4);
    draw_circle(m, 2, 2, 1, -1, color); // plus shape
    const int pts[5][2] = {{2, 1}, {1, 2}, {2, 2}, {3, 2}, {2, 3}};
    for (int i = 0; i < 5; i++) {
        memcpy(mat_row(e, pts[i][1]) + 4 * pts[i][0], color, 4);
    }
    CHECK_MAT_BYTES(m, e->data);
    struct Mat *g  = canvas(5, 5, CV_8UC1);
    struct Mat *ge = canvas(5, 5, CV_8UC1);
    draw_circle(g, 2, 2, 1, -1, color);
    for (int i = 0; i < 5; i++) {
        mat_row(ge, pts[i][1])[pts[i][0]] = 7;
    }
    CHECK_MAT_BYTES(g, ge->data);
    free_mat(&m);
    free_mat(&e);
    free_mat(&g);
    free_mat(&ge);
}

// ---- Text -------------------------------------------------------------------

static void test_text_hand(void) {
    static const char *const glyph_i[] = {
        ".###.....", "..#......", "..#......", "..#......",
        "..#......", "..#......", ".###.....", ".........",
    };
    struct Mat *m = zeros(8, 9);
    draw_text(m, "I", 0, 0, 1, WHITE);
    check_art(m, glyph_i);

    // '\n', a lone 0xC3 lead byte and a UTF-8 character (3 bytes) each draw
    // one box; a lone continuation byte draws nothing and takes no space.
    static const char *const boxes[] = {
        "#####.#####.#####.", "#...#.#...#.#...#.", "#...#.#...#.#...#.",
        "#...#.#...#.#...#.", "#...#.#...#.#...#.", "#...#.#...#.#...#.",
        "#####.#####.#####.", "..................",
    };
    struct Mat *b = zeros(8, 18);
    draw_text(b, "\n\x80\xC3\xE4\xB8\xAD", 0, 0, 1, WHITE);
    check_art(b, boxes);
    free_mat(&m);
    free_mat(&b);
}

static void test_text_width(void) {
    CHECK_EQ_INT(draw_text_width("", 1), 0);
    CHECK_EQ_INT(draw_text_width("ab", 1), 12);
    CHECK_EQ_INT(draw_text_width("ab", 3), 36);
    CHECK_EQ_INT(draw_text_width("\xE4\xB8\xAD", 2), 12); // one character
    CHECK_EQ_INT(draw_text_width("\x80\xBF", 1), 0);
    CHECK_EQ_INT(draw_text_width("x", DRAW_TEXT_MAX_SCALE),
                 6 * DRAW_TEXT_MAX_SCALE);
}

// Every printable glyph: space is empty, every other glyph has ink, no glyph
// touches the spacing column, and no two glyphs are the same.
static void test_text_font(void) {
    static uchar glyphs[95][8 * 6];
    struct Mat  *m = zeros(8, 6);
    for (int c = 0x20; c <= 0x7E; c++) {
        const char s[2] = {(char)c, 0};
        memset(m->data, 0, (size_t)8 * m->step[0]);
        draw_text(m, s, 0, 0, 1, WHITE);
        int ink = count_nonzero(m), spacing = 0;
        for (int y = 0; y < 8; y++) {
            spacing += px(m, 5, y) != 0;
        }
        CHECK(c == ' ' ? ink == 0 : ink > 0);
        CHECK_EQ_INT(spacing, 0);
        memcpy(glyphs[c - 0x20], m->data, sizeof glyphs[0]);
    }
    for (int i = 0; i < 95; i++) {
        for (int j = i + 1; j < 95; j++) {
            if (memcmp(glyphs[i], glyphs[j], sizeof glyphs[0]) == 0) {
                TEST_FAIL_("glyphs 0x%02X and 0x%02X are equal", i + 0x20,
                           j + 0x20);
            }
        }
    }
    free_mat(&m);
}

// Scale s is the scale-1 image with every pixel blown up to s x s, and a text
// drawn on a window equals the crop of the same text on a larger canvas.
static void test_text_scale_and_clip(void) {
    const char *text = "Hello, {draw} 0123 ~|_gjpqy";
    const int   w1   = draw_text_width(text, 1);
    struct Mat *one  = zeros(8, w1);
    draw_text(one, text, 0, 0, 1, WHITE);
    for (int s = 2; s <= 3; s++) {
        struct Mat *big = zeros(8 * s, w1 * s);
        draw_text(big, text, 0, 0, s, WHITE);
        int bad = 0;
        for (int y = 0; y < 8 * s && !bad; y++) {
            for (int x = 0; x < w1 * s; x++) {
                if (px(big, x, y) != px(one, x / s, y / s)) {
                    TEST_FAIL_("scale %d pixel (%d, %d)", s, x, y);
                    bad = 1;
                    break;
                }
            }
        }
        free_mat(&big);
    }
    struct Mat *canvas2 = zeros(40, 200);
    struct Mat *win     = zeros(9, 23);
    struct Mat *cut     = zeros(9, 23);
    for (int trial = 0; trial < 50; trial++) {
        const int tx = (int)rnd(-60, 150), ty = (int)rnd(-20, 35);
        const int ox = (int)rnd(0, 177), oy = (int)rnd(0, 31);
        const int s = (int)rnd(1, 3);
        memset(canvas2->data, 0, (size_t)40 * canvas2->step[0]);
        memset(win->data, 0, (size_t)9 * win->step[0]);
        draw_text(canvas2, text, tx, ty, s, WHITE);
        draw_text(win, text, tx - ox, ty - oy, s, WHITE);
        crop(cut, canvas2, ox, oy);
        CHECK(same_image(win, cut));
    }
    // Far outside in every direction: nothing drawn.
    struct Mat *e = zeros(10, 10);
    draw_text(e, text, INT32_MIN, 0, DRAW_TEXT_MAX_SCALE, WHITE);
    draw_text(e, text, INT32_MAX, 0, 1, WHITE);
    draw_text(e, text, 0, INT32_MIN, DRAW_TEXT_MAX_SCALE, WHITE);
    draw_text(e, text, 0, INT32_MAX, 1, WHITE);
    draw_text(e, text, 0, -8, 1, WHITE);
    draw_text(e, text, -6 * 27, 0, 1, WHITE); // ends exactly at column 0
    CHECK_EQ_INT(count_nonzero(e), 0);
    free_mat(&one);
    free_mat(&canvas2);
    free_mat(&win);
    free_mat(&cut);
    free_mat(&e);
}

static void test_text_channels(void) {
    const uchar color[4] = {1, 2, 3, 4};
    struct Mat *m        = canvas(8, 6, CV_8UC3);
    struct Mat *e        = canvas(8, 6, CV_8UC3);
    draw_text(m, "|", 0, 0, 1, color); // column 2, rows 0..6
    for (int y = 0; y < 7; y++) {
        memcpy(mat_row(e, y) + 3 * 2, color, 3);
    }
    CHECK_MAT_BYTES(m, e->data);
    free_mat(&m);
    free_mat(&e);
}

// ---- Misuse, empty images, scratch ------------------------------------------

static void test_empty_images(void) {
    struct Mat *z = alloc_mat(0, 5, CV_8UC3);
    struct Mat *w = alloc_mat(4, 0, CV_8UC1);
    draw_point(z, 0, 0, WHITE);
    draw_line(z, 0, 0, 4, 0, 3, WHITE);
    draw_circle(w, 0, 0, 3, -1, WHITE);
    draw_circle(w, 0, 0, 3, 1, WHITE);
    draw_text(z, "abc", 0, 0, 1, WHITE);
    free_mat(&z);
    free_mat(&w);
}

static void test_misuse(void) {
    struct Mat *g  = zeros(4, 4);
    struct Mat *c2 = alloc_mat(4, 4, CV_8UC2);
    struct Mat *s  = alloc_mat(4, 4, CV_16UC1);
    EXPECT_ASSERT(draw_point(NULL, 0, 0, WHITE));
    EXPECT_ASSERT(draw_point(g, 0, 0, NULL));
    EXPECT_ASSERT(draw_point(c2, 0, 0, WHITE));
    EXPECT_ASSERT(draw_point(s, 0, 0, WHITE));
    EXPECT_ASSERT(draw_line(g, 0, 0, 1, 1, 0, WHITE));
    EXPECT_ASSERT(draw_line(g, 0, 0, 1, 1, -1, WHITE));
    EXPECT_ASSERT(draw_line(g, 0, 0, 1, 1, DRAW_MAX_THICKNESS + 1, WHITE));
    EXPECT_ASSERT(draw_line(c2, 0, 0, 1, 1, 1, WHITE));
    EXPECT_ASSERT(draw_line(g, 0, 0, 1, 1, 1, NULL));
    EXPECT_ASSERT(draw_circle(g, 1, 1, -1, 1, WHITE));
    EXPECT_ASSERT(draw_circle(g, 1, 1, 1, 0, WHITE));
    EXPECT_ASSERT(draw_circle(g, 1, 1, 1, DRAW_MAX_THICKNESS + 1, WHITE));
    EXPECT_ASSERT(draw_circle(s, 1, 1, 1, 1, WHITE));
    EXPECT_ASSERT(draw_text(g, NULL, 0, 0, 1, WHITE));
    EXPECT_ASSERT(draw_text(g, "a", 0, 0, 0, WHITE));
    EXPECT_ASSERT(draw_text(g, "a", 0, 0, DRAW_TEXT_MAX_SCALE + 1, WHITE));
    EXPECT_ASSERT(draw_text(c2, "a", 0, 0, 1, WHITE));
    EXPECT_ASSERT(draw_text(g, "a", 0, 0, 1, NULL));
    EXPECT_ASSERT(draw_text_width(NULL, 1));
    EXPECT_ASSERT(draw_text_width("a", 0));
    CHECK_EQ_INT(count_nonzero(g), 0);
    free_mat(&g);
    free_mat(&c2);
    free_mat(&s);
}

static void test_no_scratch(void) {
    struct Mat *m = canvas(40, 50, CV_8UC3);
    alloc_counter_start();
    draw_point(m, 3, 4, WHITE);
    draw_line(m, -10, 3, 70, 30, 1, WHITE);
    draw_line(m, -10, 3, 70, 30, 6, WHITE);
    draw_circle(m, 20, 20, 15, 1, WHITE);
    draw_circle(m, 20, 20, 15, -1, WHITE);
    draw_circle(m, 20, 20, 15, 4, WHITE);
    draw_text(m, "abc", 1, 1, 2, WHITE);
    (void)draw_text_width("abc", 2);
    alloc_counter_stop();
    CHECK_EQ_INT(alloc_stats.allocs, 0);
    CHECK_EQ_INT(alloc_stats.peak, 0);
    free_mat(&m);
}

int main(void) {
    RUN_TEST(test_point_channels);
    RUN_TEST(test_line_hand);
    RUN_TEST(test_line_channels);
    RUN_TEST(test_line_sweep);
    RUN_TEST(test_line_clipping);
    RUN_TEST(test_line_huge);
    RUN_TEST(test_thick_line_hand);
    RUN_TEST(test_thick_line_sweep);
    RUN_TEST(test_thick_line_max);
    RUN_TEST(test_circle_hand);
    RUN_TEST(test_circle_sweep);
    RUN_TEST(test_circle_window);
    RUN_TEST(test_circle_huge);
    RUN_TEST(test_circle_channels);
    RUN_TEST(test_text_hand);
    RUN_TEST(test_text_width);
    RUN_TEST(test_text_font);
    RUN_TEST(test_text_scale_and_clip);
    RUN_TEST(test_text_channels);
    RUN_TEST(test_empty_images);
    RUN_TEST(test_misuse);
    RUN_TEST(test_no_scratch);
    return test_finish();
}
