// Argument guards shared by every op: a NULL pointer is misuse and ASSERTs,
// and an empty Mat (rows == 0 or cols == 0) is legal even with data == NULL,
// so every op returns without touching it.
#include <stdint.h>
#include <string.h>

#include "improc.h"
#include "test_expect_assert.h"
#include "test_mat_util.h"

// EXPECT_NO_ASSERT(stmt) runs stmt with the same longjmp handler, so an
// unexpected ASSERT is reported as a failure of this check instead of
// aborting the whole test program.
#define EXPECT_NO_ASSERT(stmt)                                                 \
    do {                                                                       \
        expect_assert_count = 0;                                               \
        improc_set_assert_handler(expect_assert_handler);                      \
        if (setjmp(expect_assert_jump) == 0) {                                 \
            stmt;                                                              \
        }                                                                      \
        improc_set_assert_handler(NULL);                                       \
        if (expect_assert_count != 0) {                                        \
            TEST_FAIL_("unexpected ASSERT from %s", #stmt);                    \
        } else {                                                               \
            CHECK(1);                                                          \
        }                                                                      \
    } while (0)

static const uchar WHITE[4] = {255, 255, 255, 255};

static void test_filter_null(void) {
    struct Mat *a = alloc_mat(8, 8, CV_8UC1);
    struct Mat *b = alloc_mat(8, 8, CV_8UC1);
    struct Mat *s = alloc_mat(8, 8, CV_16SC1);
    fill_pattern(a, 1);
    EXPECT_ASSERT(box_filter(NULL, a, 3));
    EXPECT_ASSERT(box_filter(b, NULL, 3));
    EXPECT_ASSERT(gaussian_blur(NULL, a, 3));
    EXPECT_ASSERT(gaussian_blur(b, NULL, 3));
    EXPECT_ASSERT(sobel(NULL, a, 1, 0));
    EXPECT_ASSERT(sobel(s, NULL, 1, 0));
    EXPECT_ASSERT(median_blur(NULL, a, 3));
    EXPECT_ASSERT(median_blur(b, NULL, 3));
    free_mat(&s);
    free_mat(&b);
    free_mat(&a);
}

static void test_pixel_null(void) {
    uchar       lut[256] = {0};
    struct Mat *a        = alloc_mat(4, 4, CV_8UC1);
    struct Mat *b        = alloc_mat(4, 4, CV_8UC1);
    fill_pattern(a, 2);
    EXPECT_ASSERT(threshold(NULL, a, 10, 255, THRESH_BINARY));
    EXPECT_ASSERT(threshold(b, NULL, 10, 255, THRESH_BINARY));
    EXPECT_ASSERT(mat_lut(NULL, a, lut));
    EXPECT_ASSERT(mat_lut(b, NULL, lut));
    EXPECT_ASSERT(convert_scale(NULL, a, 256, 0));
    EXPECT_ASSERT(convert_scale(b, NULL, 256, 0));
    free_mat(&b);
    free_mat(&a);
}

static void test_color_geometry_null(void) {
    struct Mat *rgb  = alloc_mat(4, 4, CV_8UC3);
    struct Mat *gray = alloc_mat(4, 4, CV_8UC1);
    struct Mat *g2   = alloc_mat(4, 4, CV_8UC1);
    struct Mat *half = alloc_mat(2, 2, CV_8UC1);
    fill_pattern(rgb, 3);
    fill_pattern(gray, 4);
    EXPECT_ASSERT(cvt_color(NULL, rgb, COLOR_RGB2GRAY));
    EXPECT_ASSERT(cvt_color(gray, NULL, COLOR_RGB2GRAY));
    EXPECT_ASSERT(flip(NULL, gray, FLIP_HORIZONTAL));
    EXPECT_ASSERT(flip(g2, NULL, FLIP_HORIZONTAL));
    EXPECT_ASSERT(resize(NULL, gray, INTER_LINEAR));
    EXPECT_ASSERT(resize(half, NULL, INTER_LINEAR));
    free_mat(&half);
    free_mat(&g2);
    free_mat(&gray);
    free_mat(&rgb);
}

#if IMPROC_HAS_FLOAT
static void test_convert_to_null(void) {
    struct Mat *a = alloc_mat(4, 4, CV_8UC1);
    struct Mat *f = alloc_mat(4, 4, CV_32FC1);
    fill_pattern(a, 5);
    EXPECT_ASSERT(convert_to(NULL, a, 1.0f, 0.0f));
    EXPECT_ASSERT(convert_to(f, NULL, 1.0f, 0.0f));
    free_mat(&f);
    free_mat(&a);
}
#endif

static void test_mat_accessors_null(void) {
    EXPECT_ASSERT((void)mat_depth(NULL));
    EXPECT_ASSERT((void)mat_channel_size(NULL));
    EXPECT_ASSERT((void)mat_channel_num(NULL));
    EXPECT_ASSERT((void)mat_pixel_size(NULL));
    EXPECT_ASSERT((void)mat_type(NULL));
}

static void test_hamming_null(void) {
    uint8_t a[ORB_DESC_BYTES] = {0};
    uint8_t b[ORB_DESC_BYTES] = {0};
    b[3]                      = 0xF1;
    CHECK_EQ_INT(hamming_distance(a, b), 5);
    EXPECT_ASSERT((void)hamming_distance(NULL, b));
    EXPECT_ASSERT((void)hamming_distance(a, NULL));
}

static void test_synthetic_pixel_guards(void) {
    uchar out[3] = {0};
    mat_synthetic_pixel(4, 4, 3, 3, out);
    CHECK_EQ_INT(out[0], 255);
    EXPECT_ASSERT(mat_synthetic_pixel(4, 4, 0, 0, NULL));
    EXPECT_ASSERT(mat_synthetic_pixel(4, 4, -1, 0, out));
    EXPECT_ASSERT(mat_synthetic_pixel(4, 4, 4, 0, out));
    EXPECT_ASSERT(mat_synthetic_pixel(4, 4, 0, -1, out));
    EXPECT_ASSERT(mat_synthetic_pixel(4, 4, 0, 4, out));
}

static void test_more_ops_null(void) {
    struct Mat *a   = alloc_mat(8, 8, CV_8UC1);
    struct Mat *b   = alloc_mat(8, 8, CV_8UC1);
    struct Mat *c   = alloc_mat(8, 8, CV_8UC1);
    struct Mat *s16 = alloc_mat(8, 8, CV_16SC1);
    fill_pattern(a, 6);
    EXPECT_ASSERT(erode(NULL, a, 3));
    EXPECT_ASSERT(erode(b, NULL, 3));
    EXPECT_ASSERT(dilate(NULL, a, 3));
    EXPECT_ASSERT(dilate(b, NULL, 3));
    EXPECT_ASSERT(morph_open(NULL, c, a, 3));
    EXPECT_ASSERT(morph_open(b, NULL, a, 3));
    EXPECT_ASSERT(morph_open(b, c, NULL, 3));
    EXPECT_ASSERT(morph_close(NULL, c, a, 3));
    EXPECT_ASSERT(morph_close(b, NULL, a, 3));
    EXPECT_ASSERT(morph_close(b, c, NULL, 3));
    EXPECT_ASSERT(absdiff(NULL, a, b));
    EXPECT_ASSERT(absdiff(c, NULL, b));
    EXPECT_ASSERT(absdiff(c, a, NULL));
    EXPECT_ASSERT(convert_scale_abs(NULL, s16, 256, 0));
    EXPECT_ASSERT(convert_scale_abs(b, NULL, 256, 0));
    EXPECT_ASSERT((void)otsu_threshold(NULL));
    struct NonzeroStats st;
    EXPECT_ASSERT(nonzero_stats(&st, NULL));
    struct KeyPoint kps[4];
    uint8_t         desc[1][ORB_DESC_BYTES];
    EXPECT_ASSERT((void)orb_detect(kps, 4, NULL, 20));
    EXPECT_ASSERT(orb_compute(desc, kps, 0, NULL));
    EXPECT_ASSERT(draw_point(NULL, 0, 0, WHITE));
    EXPECT_ASSERT(draw_line(NULL, 0, 0, 3, 3, 1, WHITE));
    EXPECT_ASSERT(draw_circle(NULL, 2, 2, 1, 1, WHITE));
    EXPECT_ASSERT(draw_text(NULL, "A", 0, 0, 1, WHITE));
    free_mat(&s16);
    free_mat(&c);
    free_mat(&b);
    free_mat(&a);
}

// The data alias checks moved after the empty return; on a non-empty image
// two Mat headers that share one buffer must still ASSERT.
static void test_alias_still_asserts(void) {
    struct Mat *a = alloc_mat(8, 8, CV_8UC1);
    struct Mat *c = alloc_mat(8, 8, CV_8UC1);
    fill_pattern(a, 7);
    struct Mat v;
    CHECK_EQ_INT(mat_init(&v, 8, 8, CV_8UC1, a->data, 64), 0);
    EXPECT_ASSERT(box_filter(&v, a, 3));
    EXPECT_ASSERT(median_blur(&v, a, 3));
    EXPECT_ASSERT(erode(&v, a, 3));
    EXPECT_ASSERT(morph_open(c, &v, a, 3));
    EXPECT_ASSERT(morph_open(&v, c, a, 3));
    EXPECT_ASSERT(equalize_hist(&v, a));
    EXPECT_ASSERT(flip(&v, a, FLIP_HORIZONTAL));
    EXPECT_ASSERT(threshold(&v, a, 10, 255, THRESH_BINARY));
    EXPECT_ASSERT(absdiff(&v, a, c));
    EXPECT_ASSERT(absdiff(&v, c, a));
    uchar lut[256] = {0};
    EXPECT_ASSERT(mat_lut(&v, a, lut));
    EXPECT_ASSERT(convert_scale(&v, a, 256, 0));
    EXPECT_ASSERT(crop(&v, a, 0, 0));
    EXPECT_ASSERT(resize(&v, a, INTER_NEAREST));

    struct Mat *rgb = alloc_mat(8, 8, CV_8UC3);
    struct Mat  v3;
    CHECK_EQ_INT(mat_init(&v3, 8, 8, CV_8UC3, rgb->data, 192), 0);
    EXPECT_ASSERT(cvt_color(&v3, rgb, COLOR_RGB2BGR));

    struct Mat *s16 = alloc_mat(8, 8, CV_16SC1);
    memset(s16->data, 0, 128);
    struct Mat v8;
    CHECK_EQ_INT(mat_init(&v8, 8, 8, CV_8UC1, s16->data, 64), 0);
    EXPECT_ASSERT(convert_scale_abs(&v8, s16, 256, 0));
#if IMPROC_HAS_FLOAT
    struct Mat *f = alloc_mat(8, 8, CV_32FC1);
    struct Mat  vf;
    CHECK_EQ_INT(mat_init(&vf, 8, 8, CV_8UC1, f->data, 64), 0);
    memset(f->data, 0, 256);
    EXPECT_ASSERT(convert_to(&vf, f, 1.0f, 0.0f));
    free_mat(&f);
#endif
    free_mat(&s16);
    free_mat(&rgb);
    free_mat(&c);
    free_mat(&a);
}

// An empty image made by mat_init with data == NULL.
static struct Mat empty_mat(int rows, int cols, int type) {
    struct Mat m;
    CHECK_EQ_INT(mat_init(&m, rows, cols, type, NULL, 0), 0);
    CHECK(m.data == NULL);
    return m;
}

static void run_empty(int rows, int cols) {
    struct Mat a   = empty_mat(rows, cols, CV_8UC1);
    struct Mat b   = empty_mat(rows, cols, CV_8UC1);
    struct Mat c   = empty_mat(rows, cols, CV_8UC1);
    struct Mat s16 = empty_mat(rows, cols, CV_16SC1);
    struct Mat c3  = empty_mat(rows, cols, CV_8UC3);
    struct Mat c3b = empty_mat(rows, cols, CV_8UC3);
    uchar      lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = (uchar)(255 - i);
    }
    uint32_t hist[256];
    memset(hist, 0xAB, sizeof(hist));

    EXPECT_NO_ASSERT(box_filter(&b, &a, 3));
    EXPECT_NO_ASSERT(gaussian_blur(&b, &a, 5));
    EXPECT_NO_ASSERT(sobel(&s16, &a, 1, 0));
    EXPECT_NO_ASSERT(median_blur(&b, &a, 3));
    EXPECT_NO_ASSERT(median_blur(&b, &a, 7));
    EXPECT_NO_ASSERT(erode(&b, &a, 3));
    EXPECT_NO_ASSERT(dilate(&b, &a, 3));
    EXPECT_NO_ASSERT(morph_open(&b, &c, &a, 3));
    EXPECT_NO_ASSERT(morph_close(&b, &c, &a, 3));

    EXPECT_NO_ASSERT(calc_hist(hist, &a));
    for (int i = 0; i < 256; i++) {
        CHECK_EQ_INT(hist[i], 0);
    }
    struct NonzeroStats st;
    EXPECT_NO_ASSERT(nonzero_stats(&st, &a));
    CHECK_EQ_INT(st.count, 0);
    EXPECT_NO_ASSERT(equalize_hist(&b, &a));
    EXPECT_NO_ASSERT((void)otsu_threshold(&a));

    EXPECT_NO_ASSERT(cvt_color(&a, &c3, COLOR_RGB2GRAY));
    EXPECT_NO_ASSERT(cvt_color(&c3b, &c3, COLOR_RGB2BGR));
    EXPECT_NO_ASSERT(cvt_color(&c3b, &c3, COLOR_RGB2HSV));
    // YUYV needs an even width; rounding 5 up to 6 keeps the image empty.
    struct Mat y2 = empty_mat(rows, (cols + 1) & ~1, CV_8UC2);
    struct Mat y3 = empty_mat(rows, (cols + 1) & ~1, CV_8UC3);
    EXPECT_NO_ASSERT(cvt_color(&y3, &y2, COLOR_YUV2RGB_YUYV));
    EXPECT_NO_ASSERT(flip(&b, &a, FLIP_BOTH));
    EXPECT_NO_ASSERT(crop(&b, &a, 0, 0));
    EXPECT_NO_ASSERT(resize(&b, &a, INTER_NEAREST));
    EXPECT_NO_ASSERT(resize(&b, &a, INTER_LINEAR));
    EXPECT_NO_ASSERT(resize(&b, &a, INTER_AREA));

    EXPECT_NO_ASSERT(threshold(&b, &a, 10, 255, THRESH_BINARY));
    EXPECT_NO_ASSERT(mat_lut(&b, &a, lut));
    EXPECT_NO_ASSERT(absdiff(&c, &a, &b));
    EXPECT_NO_ASSERT(absdiff(&c, &a, &a));
    EXPECT_NO_ASSERT(convert_scale(&b, &a, 256, 1));
    EXPECT_NO_ASSERT(convert_scale_abs(&b, &s16, 256, 1));
#if IMPROC_HAS_FLOAT
    struct Mat f32 = empty_mat(rows, cols, CV_32FC1);
    struct Mat f2  = empty_mat(rows, cols, CV_32FC1);
    EXPECT_NO_ASSERT(convert_to(&f32, &a, 1.0f, 0.0f));
    EXPECT_NO_ASSERT(convert_to(&b, &f32, 1.0f, 0.0f));
    EXPECT_NO_ASSERT(convert_to(&f2, &f32, 2.0f, 0.0f));
    struct SiftKeyPoint skp[2];
    uint8_t             sdesc[2][SIFT_DESC_BYTES];
    int32_t             ns = -1;
    EXPECT_NO_ASSERT(ns = sift_detect_compute(skp, sdesc, 2, &a, NULL));
    CHECK_EQ_INT(ns, 0);
    ns = -1;
    EXPECT_NO_ASSERT(ns = sift_detect(skp, 2, &a, NULL));
    CHECK_EQ_INT(ns, 0);
    EXPECT_NO_ASSERT(sift_compute(sdesc, skp, 0, &a, NULL));
#endif

    struct KeyPoint kps[4];
    int32_t         n = -1;
    EXPECT_NO_ASSERT(n = fast_detect(kps, 4, &a, 20, 1));
    CHECK_EQ_INT(n, 0);
    n = -1;
    EXPECT_NO_ASSERT(n = fast_detect(NULL, 0, &a, 20, 0));
    CHECK_EQ_INT(n, 0);
    n = -1;
    EXPECT_NO_ASSERT(n = orb_detect(kps, 4, &a, 20));
    CHECK_EQ_INT(n, 0);
    uint8_t desc[1][ORB_DESC_BYTES];
    EXPECT_NO_ASSERT(orb_compute(desc, kps, 0, &a));

    EXPECT_NO_ASSERT(draw_point(&a, 0, 0, WHITE));
    EXPECT_NO_ASSERT(draw_line(&a, -5, -5, 5, 5, 1, WHITE));
    EXPECT_NO_ASSERT(draw_line(&a, -5, -5, 5, 5, 3, WHITE));
    EXPECT_NO_ASSERT(draw_circle(&a, 0, 0, 3, -1, WHITE));
    EXPECT_NO_ASSERT(draw_text(&a, "Ab", 0, 0, 1, WHITE));
}

static void test_empty_null_data(void) {
    run_empty(0, 0);
    run_empty(5, 0);
    run_empty(0, 5);
}

int main(void) {
    RUN_TEST(test_filter_null);
    RUN_TEST(test_pixel_null);
    RUN_TEST(test_color_geometry_null);
#if IMPROC_HAS_FLOAT
    RUN_TEST(test_convert_to_null);
#endif
    RUN_TEST(test_mat_accessors_null);
    RUN_TEST(test_hamming_null);
    RUN_TEST(test_synthetic_pixel_guards);
    RUN_TEST(test_more_ops_null);
    RUN_TEST(test_alias_still_asserts);
    RUN_TEST(test_empty_null_data);
    return test_finish();
}
