// SIFT on extreme synthetic images: flat, blurred, saturated, low contrast,
// noisy and tiny. Every case prints its keypoint count. With --dump <dir>,
// each case image is written as <name>.pgm, its keypoints as <name>_kps.txt,
// and one line per case to cases.txt, for tools/sift_edge_cv.py.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "improc.h"
#include "test_util.h"

#if !IMPROC_HAS_FLOAT

int main(void) {
    (void)test_failures_current; // used only by RUN_TEST
    (void)test_skip_current;
    printf("Skipped: built with IMPROC_ENABLE_FLOAT=OFF\n");
    return TEST_EXIT_SKIP;
}

#else

#include "test_alloc_util.h"

#define EDGE_CAP 4000 // far above any count these 128 x 128 cases yield
#define EDGE_N   128

static struct SiftKeyPoint kps_a[EDGE_CAP], kps_b[EDGE_CAP];
static uint8_t             desc_a[EDGE_CAP][SIFT_DESC_BYTES];
static uint8_t             desc_b[EDGE_CAP][SIFT_DESC_BYTES];
static const char         *dump_dir = NULL;

static uint32_t lcg_state;

static uint32_t lcg(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

static struct Mat *filled(int rows, int cols, uchar v) {
    struct Mat *m = alloc_mat(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; y++) {
        memset(mat_row(m, y), v, (size_t)cols);
    }
    return m;
}

static struct Mat *noise(int rows, int cols, uint32_t seed) {
    struct Mat *m = alloc_mat(rows, cols, CV_8UC1);
    lcg_state     = seed;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            mat_row(m, y)[x] = (uchar)(lcg() >> 24);
        }
    }
    return m;
}

// Random bytes, 5x5 box blur, contrast stretched 4x about 128 and clamped.
static struct Mat *texture(void) {
    struct Mat *n = noise(EDGE_N, EDGE_N, 12345u);
    struct Mat *t = alloc_mat(EDGE_N, EDGE_N, CV_8UC1);
    box_filter(t, n, 5);
    for (int y = 0; y < EDGE_N; y++) {
        uchar *r = mat_row(t, y);
        for (int x = 0; x < EDGE_N; x++) {
            const int32_t v = 128 + ((int32_t)r[x] - 128) * 4;
            r[x]            = (uchar)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    free_mat(&n);
    return t;
}

// Pixels within radius of (64, 64) set to v.
static void paint_disc(struct Mat *m, int32_t radius, uchar v) {
    for (int32_t y = 0; y < m->rows; y++) {
        for (int32_t x = 0; x < m->cols; x++) {
            const int32_t dx = x - 64, dy = y - 64;
            if (dx * dx + dy * dy <= radius * radius) {
                mat_row(m, (int)y)[x] = v;
            }
        }
    }
}

static void dump_case(const char *name, const struct Mat *img, int32_t n) {
    if (dump_dir == NULL) {
        return;
    }
    char  path[1024];
    FILE *f;
    snprintf(path, sizeof path, "%s/%s.pgm", dump_dir, name);
    if ((f = fopen(path, "wb")) != NULL) {
        fprintf(f, "P5\n%d %d\n255\n", img->cols, img->rows);
        for (int y = 0; y < img->rows; y++) {
            fwrite(mat_row(img, y), 1, (size_t)img->cols, f);
        }
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/%s_kps.txt", dump_dir, name);
    if ((f = fopen(path, "w")) != NULL) {
        for (int32_t i = 0; i < n; i++) {
            const struct SiftKeyPoint *k = &kps_a[i];
            fprintf(f, "%.9g %.9g %.9g %.9g %.9g %ld %ld\n", (double)k->x,
                    (double)k->y, (double)k->size, (double)k->angle,
                    (double)k->response, (long)k->octave, (long)k->layer);
        }
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/cases.txt", dump_dir);
    if ((f = fopen(path, "a")) != NULL) {
        fprintf(f, "%s %d %d %ld\n", name, img->rows, img->cols, (long)n);
        fclose(f);
    }
}

// Runs SIFT with the defaults into kps_a / desc_a, checks every keypoint
// field, prints and dumps the case. The count must stay below the capacity,
// or the counts compared by the cases would be truncated.
static int32_t run_case(const char *name, const struct Mat *img) {
    const int32_t n = sift_detect_compute(kps_a, desc_a, EDGE_CAP, img, NULL);
    CHECK(n >= 0 && n < EDGE_CAP);
    for (int32_t i = 0; i < n; i++) {
        const struct SiftKeyPoint *k = &kps_a[i];
        CHECK(isfinite(k->x) && isfinite(k->y) && isfinite(k->size) &&
              isfinite(k->angle) && isfinite(k->response));
        CHECK(k->angle >= 0.0f && k->angle < 360.0f);
        CHECK(k->size > 0.0f);
        CHECK(k->x >= 0.0f && k->x <= (float)(img->cols - 1));
        CHECK(k->y >= 0.0f && k->y <= (float)(img->rows - 1));
    }
    printf("  %-14s %4dx%-4d keypoints %ld\n", name, img->rows, img->cols,
           (long)n);
    dump_case(name, img, n);
    return n;
}

static void test_flat(void) {
    static const uchar levels[3] = {0, 255, 128};
    static const char *names[3]  = {"flat_0", "flat_255", "flat_128"};
    for (int i = 0; i < 3; i++) {
        struct Mat *m = filled(EDGE_N, EDGE_N, levels[i]);
        CHECK_EQ_INT(run_case(names[i], m), 0);
        free_mat(&m);
    }
}

static void test_extreme_blur(void) {
    struct Mat *a = filled(EDGE_N, EDGE_N, 0);
    struct Mat *b = alloc_mat(EDGE_N, EDGE_N, CV_8UC1);
    paint_disc(a, 30, 255);
    const int32_t sharp = run_case("disc", a);
    // 30 passes of a 31 x 31 box: Gaussian-like, sigma about 49 pixels.
    for (int i = 0; i < 15; i++) {
        box_filter(b, a, 31);
        box_filter(a, b, 31);
    }
    int lo = 255, hi = 0;
    for (int y = 0; y < EDGE_N; y++) {
        for (int x = 0; x < EDGE_N; x++) {
            const int v = mat_row(a, y)[x];
            lo          = v < lo ? v : lo;
            hi          = v > hi ? v : hi;
        }
    }
    const int32_t blurred = run_case("disc_blur", a);
    printf("  disc %ld keypoints, blurred %ld (blurred range %d..%d)\n",
           (long)sharp, (long)blurred, lo, hi);
    CHECK(sharp > 0);
    CHECK(blurred <= sharp);
    free_mat(&a);
    free_mat(&b);
}

static void test_glare(void) {
    struct Mat   *m     = texture();
    const int32_t plain = run_case("texture", m);
    paint_disc(m, 25, 255);
    const int32_t n = run_case("glare", m);
    printf("  texture %ld keypoints, with glare %ld\n", (long)plain, (long)n);
    // A keypoint may sit inside the saturated disc only as far as its own
    // neighbourhood (size / 2) reaches out to the texture, plus 4 pixels.
    // Exception: the disc itself is a bright blob, found at its centre with
    // sigma near radius / sqrt(2) (size 32.8, four orientations; cv2.SIFT
    // finds the same four). Keypoints at least as large as the radius must
    // lie at the centre instead.
    int32_t whole = 0;
    for (int32_t i = 0; i < n; i++) {
        const struct SiftKeyPoint *k  = &kps_a[i];
        const float                dx = k->x - 64.0f, dy = k->y - 64.0f;
        const float                dist  = sqrtf(dx * dx + dy * dy);
        const float                depth = 25.0f - dist;
        if (k->size >= 25.0f && depth > 0.0f) {
            CHECK(dist <= 1.5f);
            whole++;
            continue;
        }
        if (depth > 4.0f + k->size * 0.5f) {
            printf("  deep in glare: x %.2f y %.2f size %.2f depth %.2f\n",
                   (double)k->x, (double)k->y, (double)k->size, (double)depth);
            CHECK(depth <= 4.0f + k->size * 0.5f);
        }
    }
    printf("  %ld keypoints of the disc as a whole, at its centre\n",
           (long)whole);
    CHECK(whole <= 4);
    free_mat(&m);
}

static void test_low_contrast(void) {
    struct Mat *m = texture();
    // The full-contrast image is the "texture" case of test_glare.
    const int32_t full = sift_detect_compute(kps_a, NULL, EDGE_CAP, m, NULL);
    CHECK(full < EDGE_CAP);
    for (int y = 0; y < EDGE_N; y++) {
        uchar *r = mat_row(m, y);
        for (int x = 0; x < EDGE_N; x++) {
            r[x] = (uchar)(126 + ((int32_t)r[x] * 4 + 127) / 255);
        }
    }
    const int32_t low = run_case("low_contrast", m);
    printf("  full contrast %ld keypoints, range 126..130 %ld\n", (long)full,
           (long)low);
    CHECK(low < full);
    free_mat(&m);
}

static void test_salt_pepper(void) {
    struct Mat *m = texture();
    lcg_state     = 777u;
    for (int y = 0; y < EDGE_N; y++) {
        for (int x = 0; x < EDGE_N; x++) {
            const uint32_t r = (lcg() >> 16) % 100u;
            if (r < 5u) {
                mat_row(m, y)[x] = (r < 3u) ? 0 : 255; // about 2.5 % each
            }
        }
    }
    run_case("salt_pepper", m);
    free_mat(&m);
}

static void test_tiny(void) {
    static const int sizes[][2] = {{0, 0}, {1, 1},   {5, 5},   {6, 6},
                                   {7, 7}, {12, 12}, {1, 200}, {200, 1}};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        const int   rows = sizes[i][0], cols = sizes[i][1];
        struct Mat *m = noise(rows, cols, 99u + (uint32_t)i);
        char        name[32];
        snprintf(name, sizeof name, "tiny_%dx%d", rows, cols);
        const size_t want = sift_scratch_bytes(rows, cols, NULL);
        alloc_counter_start();
        const int32_t n = run_case(name, m);
        alloc_counter_stop();
        printf("    scratch %zu bytes, allocated %zu in %ld blocks\n", want,
               alloc_stats.peak, alloc_stats.allocs);
        if (want == 0) {
            CHECK_EQ_INT(n, 0);
            CHECK_EQ_INT(alloc_stats.allocs, 0);
        } else {
            CHECK(alloc_stats.peak >= want && alloc_stats.peak <= want + 64);
            CHECK_EQ_INT(alloc_stats.allocs, 1);
        }
        free_mat(&m);
    }
}

static void test_capacity(void) {
    struct Mat   *m   = texture();
    const int32_t all = sift_detect_compute(kps_b, desc_b, EDGE_CAP, m, NULL);
    CHECK(all > 1 && all < EDGE_CAP);
    CHECK_EQ_INT(sift_detect_compute(kps_a, desc_a, 0, m, NULL), 0);
    CHECK_EQ_INT(sift_detect_compute(NULL, NULL, 0, m, NULL), 0);
    const int32_t one = sift_detect_compute(kps_a, desc_a, 1, m, NULL);
    CHECK_EQ_INT(one, 1);
    // The single keypoint kept is the strongest one of the full run.
    CHECK_EQ_BYTES(&kps_a[0], &kps_b[0], sizeof kps_a[0]);
    CHECK_EQ_BYTES(desc_a[0], desc_b[0], SIFT_DESC_BYTES);
    printf("  capacity 0 -> 0, capacity 1 -> %ld (of %ld)\n", (long)one,
           (long)all);
    free_mat(&m);
}

static void test_determinism(void) {
    struct Mat *m = texture();
    paint_disc(m, 25, 255);
    memset(kps_a, 0, sizeof kps_a);
    memset(kps_b, 0, sizeof kps_b);
    const int32_t n1 = sift_detect_compute(kps_a, desc_a, EDGE_CAP, m, NULL);
    const int32_t n2 = sift_detect_compute(kps_b, desc_b, EDGE_CAP, m, NULL);
    CHECK_EQ_INT(n2, n1);
    CHECK_EQ_BYTES(kps_b, kps_a, (size_t)n1 * sizeof kps_a[0]);
    CHECK_EQ_BYTES(desc_b, desc_a, (size_t)n1 * SIFT_DESC_BYTES);
    memset(kps_b, 0, sizeof kps_b);
    const int32_t n3 = sift_detect_compute(kps_b, NULL, EDGE_CAP, m, NULL);
    CHECK_EQ_INT(n3, n1);
    CHECK_EQ_BYTES(kps_b, kps_a, (size_t)n1 * sizeof kps_a[0]);
    printf("  %ld keypoints, identical on rerun and with desc NULL\n",
           (long)n1);
    free_mat(&m);
}

int main(int argc, char **argv) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0) {
            dump_dir = argv[i + 1];
        }
    }
    if (dump_dir != NULL) {
        char  path[1024];
        FILE *f;
        snprintf(path, sizeof path, "%s/cases.txt", dump_dir);
        if ((f = fopen(path, "w")) == NULL) {
            printf("cannot write %s\n", path);
            return 1;
        }
        fclose(f);
    }
    RUN_TEST(test_flat);
    RUN_TEST(test_extreme_blur);
    RUN_TEST(test_glare);
    RUN_TEST(test_low_contrast);
    RUN_TEST(test_salt_pepper);
    RUN_TEST(test_tiny);
    RUN_TEST(test_capacity);
    RUN_TEST(test_determinism);
    return test_finish();
}

#endif // IMPROC_HAS_FLOAT
