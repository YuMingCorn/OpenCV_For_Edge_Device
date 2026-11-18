#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "improc.h"
#include "test_util.h"

#if !IMPROC_HAS_FLOAT

int main(void) {
    printf("Skipped: built with IMPROC_ENABLE_FLOAT=OFF\n");
    return TEST_EXIT_SKIP;
}

#else

#include "test_alloc_util.h"
#include "test_expect_assert.h"

// The private helpers (exp_f, atan2_deg, gauss_blur, ...) are tested
// directly. This copy of sift.c also provides the public functions, so the
// library's sift.o is not linked into this test.
#include "sift.c"

static uint32_t rng_state = 2463534242u;

static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

// Smooth random texture: random bytes blurred with a 5x5 box, so SIFT finds
// blob-like structure at several scales.
static struct Mat *texture(int rows, int cols, uint32_t seed) {
    struct Mat *noise = alloc_mat(rows, cols, CV_8UC1);
    struct Mat *out   = alloc_mat(rows, cols, CV_8UC1);
    rng_state         = seed;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            mat_row(noise, y)[x] = (uchar)(xorshift32() >> 24);
        }
    }
    if (rows > 0 && cols > 0) {
        box_filter(out, noise, 5);
    }
    free_mat(&noise);
    return out;
}

// ---- Portable math ----------------------------------------------------------

static void test_exp_accuracy(void) {
    double worst = 0.0;
    for (int i = 0; i <= 92000; i++) {
        const float  x   = -87.0f + (float)i * 0.001f;
        const double ref = exp((double)x);
        const double rel = fabs((double)exp_f(x) - ref) / ref;
        worst            = rel > worst ? rel : worst;
    }
    CHECK(worst < 3e-7);
    CHECK(exp_f(0.0f) == 1.0f);
    CHECK(exp_f(-88.0f) == 0.0f);
    CHECK(exp_f(-1000.0f) == 0.0f);
}

static void test_atan2_accuracy(void) {
    double worst = 0.0;
    for (int iy = -200; iy <= 200; iy++) {
        for (int ix = -200; ix <= 200; ix++) {
            if (ix == 0 && iy == 0) {
                continue;
            }
            const float y = (float)iy * 0.37f, x = (float)ix * 0.37f;
            const float d = atan2_deg(y, x);
            CHECK(d >= 0.0f && d < 360.0f);
            double ref = atan2((double)y, (double)x) * 180.0 / 3.14159265358979;
            if (ref < 0.0) {
                ref += 360.0;
            }
            double err = fabs((double)d - ref);
            err        = err > 180.0 ? 360.0 - err : err;
            worst      = err > worst ? err : worst;
        }
    }
    CHECK(worst < 1e-4);
    CHECK(atan2_deg(0.0f, 0.0f) == 0.0f);
    CHECK(atan2_deg(0.0f, 1.0f) == 0.0f);
    CHECK(fabsf(atan2_deg(1.0f, 0.0f) - 90.0f) < 1e-4f);
    CHECK(fabsf(atan2_deg(0.0f, -1.0f) - 180.0f) < 1e-4f);
    CHECK(fabsf(atan2_deg(-1.0f, 0.0f) - 270.0f) < 1e-4f);
    // Just below the +x axis: close to 360 but never 360 itself.
    CHECK(atan2_deg(-1e-30f, 1.0f) < 360.0f);
}

static void test_sincos_accuracy(void) {
    double worst = 0.0;
    for (int i = 0; i < 720 * 16; i++) {
        const float deg = (float)i / 16.0f;
        float       s, c;
        sincos_deg(deg, &s, &c);
        const double rad = (double)deg * 3.14159265358979 / 180.0;
        const double es  = fabs((double)s - sin(rad));
        const double ec  = fabs((double)c - cos(rad));
        worst            = es > worst ? es : worst;
        worst            = ec > worst ? ec : worst;
    }
    CHECK(worst < 1e-6);
}

static void test_round_floor(void) {
    CHECK_EQ_INT(floor_i(2.5f), 2);
    CHECK_EQ_INT(floor_i(-2.5f), -3);
    CHECK_EQ_INT(floor_i(-3.0f), -3);
    CHECK_EQ_INT(round_i(2.5f), 3);
    CHECK_EQ_INT(round_i(-2.5f), -2);
    CHECK_EQ_INT(round_i(-2.6f), -3);
    CHECK(pow2_i(-126) == ldexpf(1.0f, -126));
    CHECK(pow2_i(0) == 1.0f && pow2_i(5) == 32.0f);
}

// ---- Scale space ------------------------------------------------------------

static int32_t reflect_ref(int32_t i, int32_t n) {
    // Walks the reflection one step at a time: 101 means the edge pixel is
    // not repeated.
    while (i < 0 || i >= n) {
        if (n == 1) {
            return 0;
        }
        i = i < 0 ? -i : 2 * (n - 1) - i;
    }
    return i;
}

static void test_reflect101(void) {
    for (int32_t n = 1; n <= 9; n++) {
        for (int32_t i = -40; i < 40; i++) {
            CHECK_EQ_INT(reflect101(i, n), reflect_ref(i, n));
        }
    }
}

// gauss_blur against a double-precision 2-D convolution with the same
// kernel (cv2 radius rule), widths 1..65.
static void test_blur_sweep(void) {
    static const float sigmas[]  = {0.7f, 1.6f, 3.1f};
    static const int   heights[] = {1, 2, 7, 20};
    float             *w         = malloc(64 * sizeof(float));
    for (int si = 0; si < 3; si++) {
        const float   sigma = sigmas[si];
        const int32_t r     = kernel_radius(sigma);
        double        k[64], sum = 0.0;
        for (int32_t i = -r; i <= r; i++) {
            k[i + r] =
                exp(-(double)(i * i) / (2.0 * (double)sigma * (double)sigma));
            sum += k[i + r];
        }
        for (int32_t i = 0; i <= 2 * r; i++) {
            k[i] /= sum;
        }
        for (int hi = 0; hi < 4; hi++) {
            const int32_t rows = heights[hi];
            for (int32_t cols = 1; cols <= 65; cols++) {
                const size_t n   = (size_t)rows * (size_t)cols;
                float       *buf = malloc(3 * n * sizeof(float));
                struct fimg  src = {buf, rows, cols};
                struct fimg  tmp = {buf + n, rows, cols};
                struct fimg  dst = {buf + 2 * n, 0, 0};
                for (size_t i = 0; i < n; i++) {
                    buf[i] = (float)(xorshift32() >> 24);
                }
                gauss_blur(&dst, &src, &tmp, sigma, w);
                CHECK(dst.rows == rows && dst.cols == cols);
                double worst = 0.0;
                for (int32_t y = 0; y < rows; y++) {
                    for (int32_t x = 0; x < cols; x++) {
                        double ref = 0.0;
                        for (int32_t dy = -r; dy <= r; dy++) {
                            for (int32_t dx = -r; dx <= r; dx++) {
                                ref +=
                                    k[dy + r] * k[dx + r] *
                                    (double)px(&src, reflect_ref(y + dy, rows),
                                               reflect_ref(x + dx, cols));
                            }
                        }
                        const double e = fabs((double)px(&dst, y, x) - ref);
                        worst          = e > worst ? e : worst;
                    }
                }
                if (worst > 2e-3) {
                    TEST_FAIL_("sigma %g, %dx%d: error %g", (double)sigma,
                               (int)rows, (int)cols, worst);
                }
                free(buf);
            }
        }
    }
    free(w);
}

static void test_octave_count(void) {
    // round(log2(m)) - 2, from the hand values and then against libm.
    CHECK_EQ_INT(octave_count(5, 100), 0); // log2 5 = 2.32
    CHECK_EQ_INT(octave_count(6, 6), 1);   // 2.58 -> 3
    CHECK_EQ_INT(octave_count(11, 11), 1); // 3.46 -> 3
    CHECK_EQ_INT(octave_count(12, 50), 2); // 3.58 -> 4
    CHECK_EQ_INT(octave_count(1024, 1536), 8);
    for (int32_t m = 1; m <= 5000; m++) {
        const int32_t ref = (int32_t)floor(log2((double)m) + 0.5) - 2;
        CHECK_EQ_INT(octave_count(m, m + 7), ref);
    }
}

static void test_layer_sigmas(void) {
    struct SiftParams p;
    sift_default_params(&p);
    float sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, &p);
    // Layer i must reach total blur sigma * 2^(i / s).
    double total = (double)p.sigma;
    for (int i = 1; i < p.n_octave_layers + 3; i++) {
        total = sqrt(total * total + (double)sig[i] * (double)sig[i]);
        const double ref = (double)p.sigma * pow(2.0, (double)i / 3.0);
        CHECK(fabs(total - ref) < 1e-5 * ref);
    }
}

// ---- Detection --------------------------------------------------------------

// A bright Gaussian blob of sigma 4 centred on pixel (32, 32) must give a
// keypoint there (plus cv2's quarter-pixel offset of the upsampled octaves),
// with size close to 2 * 4: the scale-normalised Laplacian of a Gaussian blob
// peaks at the blob's sigma, and size is twice the scale.
static void test_blob(void) {
    struct Mat *img = alloc_mat(64, 64, CV_8UC1);
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            const double d2 =
                (double)((x - 32) * (x - 32) + (y - 32) * (y - 32));
            mat_row(img, y)[x] = (uchar)(20.0 + 200.0 * exp(-d2 / 32.0) + 0.5);
        }
    }
    struct SiftKeyPoint kps[64];
    const int32_t       n     = sift_detect_compute(kps, NULL, 64, img, NULL);
    int                 found = 0;
    for (int32_t i = 0; i < n; i++) {
        if (fabsf(kps[i].x - 32.25f) < 0.1f &&
            fabsf(kps[i].y - 32.25f) < 0.1f) {
            found = 1;
            CHECK(kps[i].size > 0.8f * 8.0f && kps[i].size < 1.25f * 8.0f);
        }
    }
    CHECK(found);
    free_mat(&img);
}

// Rotating the image by exactly 90 degrees moves each keypoint of the
// upsampled octave (which keeps the pixel grid symmetric) to the rotated
// position, turns its angle by 90 degrees and keeps its descriptor. The
// reported position is the true one plus 0.25 (cv2's convention for the
// upsampled octaves), so the mirror of x is (sz - 0.5) - x, not (sz - 1) - x.
static void test_rotate90(void) {
    const int   sz  = 96;
    struct Mat *img = texture(sz, sz, 12345u);
    struct Mat *rot = alloc_mat(sz, sz, CV_8UC1);
    // Clockwise on screen: (x, y) -> (sz - 1 - y, x).
    for (int y = 0; y < sz; y++) {
        for (int x = 0; x < sz; x++) {
            mat_row(rot, x)[sz - 1 - y] = mat_row(img, y)[x];
        }
    }
    enum { CAP = 2000 };
    static struct SiftKeyPoint ka[CAP], kb[CAP];
    static uint8_t da[CAP][SIFT_DESC_BYTES], db[CAP][SIFT_DESC_BYTES];
    const int32_t  na    = sift_detect_compute(ka, da, CAP, img, NULL);
    const int32_t  nb    = sift_detect_compute(kb, db, CAP, rot, NULL);
    int32_t        total = 0, matched = 0;
    for (int32_t i = 0; i < na; i++) {
        if (ka[i].octave != -1) {
            continue;
        }
        total++;
        const float ex = ((float)sz - 0.5f) - ka[i].y, ey = ka[i].x;
        float       ea = ka[i].angle + 90.0f;
        ea             = ea >= 360.0f ? ea - 360.0f : ea;
        for (int32_t j = 0; j < nb; j++) {
            float da_ = fabsf(kb[j].angle - ea);
            da_       = da_ > 180.0f ? 360.0f - da_ : da_;
            if (fabsf(kb[j].x - ex) < 0.05f && fabsf(kb[j].y - ey) < 0.05f &&
                fabsf(kb[j].size - ka[i].size) < 0.01f * ka[i].size &&
                da_ < 1.0f) {
                int32_t d2 = 0;
                for (int k = 0; k < SIFT_DESC_BYTES; k++) {
                    const int32_t d = (int32_t)da[i][k] - (int32_t)db[j][k];
                    d2 += d * d;
                }
                matched += d2 <= 4 * SIFT_DESC_BYTES; // about 2 per byte
                break;
            }
        }
    }
    CHECK(total >= 20);
    CHECK(matched * 100 >= total * 90);
    if (matched * 100 < total * 90) {
        printf("  rotate90: %d of %d matched\n", (int)matched, (int)total);
    }
    free_mat(&img);
    free_mat(&rot);
}

// Every size from 0 to 65 in both directions, with and without upsample:
// no crash, nothing for images without an octave, the same answer twice,
// keypoints inside the image, sorted by response.
static void test_size_sweep(void) {
    enum { CAP = 512 };
    static struct SiftKeyPoint k1[CAP], k2[CAP];
    static uint8_t    d1[CAP][SIFT_DESC_BYTES], d2[CAP][SIFT_DESC_BYTES];
    struct SiftParams p;
    sift_default_params(&p);
    for (int up = 0; up <= 1; up++) {
        p.upsample = up;
        for (int s = 0; s <= 65; s++) {
            const int     rows = s, cols = (s * 7) % 66;
            struct Mat   *img   = texture(rows, cols, 777u + (uint32_t)s);
            const int     m     = rows < cols ? rows : cols;
            const size_t  bytes = sift_scratch_bytes(rows, cols, &p);
            const int32_t n1    = sift_detect_compute(k1, d1, CAP, img, &p);
            const int32_t n2    = sift_detect_compute(k2, d2, CAP, img, &p);
            CHECK_EQ_INT(n1, n2);
            if ((up ? 2 * m : m) < 6) {
                CHECK_EQ_INT(bytes, 0);
                CHECK_EQ_INT(n1, 0);
            } else {
                const size_t r0 = (size_t)(up ? 2 * rows : rows);
                const size_t c0 = (size_t)(up ? 2 * cols : cols);
                CHECK_EQ_INT(bytes, (7 * r0 * c0 + 360 + 14) * sizeof(float));
            }
            CHECK(memcmp(k1, k2, sizeof(k1[0]) * (size_t)n1) == 0);
            CHECK(memcmp(d1, d2, SIFT_DESC_BYTES * (size_t)n1) == 0);
            for (int32_t i = 0; i < n1; i++) {
                CHECK(k1[i].x >= 0.0f && k1[i].x <= (float)(cols - 1));
                CHECK(k1[i].y >= 0.0f && k1[i].y <= (float)(rows - 1));
                CHECK(k1[i].angle >= 0.0f && k1[i].angle < 360.0f);
                CHECK(i == 0 || k1[i].response <= k1[i - 1].response);
                CHECK(k1[i].layer >= 1 && k1[i].layer <= 3);
            }
            free_mat(&img);
        }
    }
}

// A smaller capacity keeps the strongest keypoints: the prefix of the full
// result. desc NULL finds the same keypoints.
static void test_capacity_and_null_desc(void) {
    enum { CAP = 3000 };
    static struct SiftKeyPoint full[CAP], part[CAP], nodesc[CAP];
    static uint8_t df[CAP][SIFT_DESC_BYTES], dp[CAP][SIFT_DESC_BYTES];
    struct Mat    *img = texture(120, 160, 4242u);
    const int32_t  n   = sift_detect_compute(full, df, CAP, img, NULL);
    CHECK(n > 40 && n < CAP);
    for (int32_t i = 1; i < n; i++) {
        const struct SiftKeyPoint *a = &full[i - 1], *b = &full[i];
        CHECK(!(a->x == b->x && a->y == b->y && a->size == b->size &&
                a->angle == b->angle)); // copies dropped
    }
    const int32_t k = n / 3;
    CHECK_EQ_INT(sift_detect_compute(part, dp, k, img, NULL), k);
    CHECK(memcmp(part, full, sizeof(full[0]) * (size_t)k) == 0);
    CHECK(memcmp(dp, df, SIFT_DESC_BYTES * (size_t)k) == 0);
    CHECK_EQ_INT(sift_detect_compute(nodesc, NULL, CAP, img, NULL), n);
    CHECK(memcmp(nodesc, full, sizeof(full[0]) * (size_t)n) == 0);
    CHECK_EQ_INT(sift_detect_compute(NULL, NULL, 0, img, NULL), 0);
    free_mat(&img);
}

// Descriptor invariant: after clipping and scaling, the norm is 512 up to
// rounding (and the rare saturation at 255).
static void test_descriptor_norm(void) {
    enum { CAP = 500 };
    static struct SiftKeyPoint kps[CAP];
    static uint8_t             desc[CAP][SIFT_DESC_BYTES];
    struct Mat                *img = texture(100, 100, 99u);
    const int32_t n = sift_detect_compute(kps, desc, CAP, img, NULL);
    CHECK(n > 10);
    for (int32_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < SIFT_DESC_BYTES; k++) {
            sum += (double)desc[i][k] * (double)desc[i][k];
        }
        CHECK(fabs(sqrt(sum) - 512.0) < 8.0);
    }
    free_mat(&img);
}

// ---- Matching ---------------------------------------------------------------

static void test_match_hand(void) {
    static uint8_t q[3][SIFT_DESC_BYTES], t[4][SIFT_DESC_BYTES];
    memset(q, 0, sizeof q);
    memset(t, 0, sizeof t);
    // q0 = 0. t0: all 10 -> 12800. t1, t2: one byte 100 -> 10000 each (tie).
    memset(t[0], 10, SIFT_DESC_BYTES);
    t[1][5] = 100;
    t[2][9] = 100;
    // q1 = all 200. t3 = all 199 -> 128; runner-up t0 -> 128 * 190^2.
    memset(q[1], 200, SIFT_DESC_BYTES);
    memset(t[3], 199, SIFT_DESC_BYTES);
    // q2: byte 5 = 90 -> t1 100, t2 8100 + 10000, t0 6400 + 127 * 100.
    q[2][5] = 90;
    struct SiftMatch m[3];
    CHECK_EQ_INT(sift_match(m, q, 3, t, 4, 0.8f), 2);
    // q0: t1 and t2 tie for the best (d1 = d2) -> rejected.
    CHECK_EQ_INT(m[0].query, 1);
    CHECK_EQ_INT(m[0].train, 3);
    CHECK_EQ_INT(m[0].distance, 128);
    CHECK_EQ_INT(m[1].query, 2);
    CHECK_EQ_INT(m[1].train, 1);
    CHECK_EQ_INT(m[1].distance, 100);
    // Ratio 1 still rejects the exact tie of q0.
    CHECK_EQ_INT(sift_match(m, q, 1, t, 4, 1.0f), 0);
    // One train descriptor: every best match is kept.
    CHECK_EQ_INT(sift_match(m, q, 3, t, 1, 0.5f), 3);
    CHECK_EQ_INT(m[2].train, 0);
    CHECK_EQ_INT(m[0].distance, 12800);
    CHECK_EQ_INT(sift_match(m, q, 0, t, 4, 0.8f), 0);
    CHECK_EQ_INT(sift_match(NULL, NULL, 0, t, 4, 0.8f), 0);
    CHECK_EQ_INT(sift_match(m, q, 3, NULL, 0, 0.8f), 0);
}

// Brute-force reference over random descriptors.
static void test_match_random(void) {
    enum { NQ = 40, NT = 60 };
    static uint8_t q[NQ][SIFT_DESC_BYTES], t[NT][SIFT_DESC_BYTES];
    rng_state = 31337u;
    for (int i = 0; i < NQ; i++) {
        for (int k = 0; k < SIFT_DESC_BYTES; k++) {
            q[i][k] = (uint8_t)(xorshift32() >> 26);
        }
    }
    for (int i = 0; i < NT; i++) {
        for (int k = 0; k < SIFT_DESC_BYTES; k++) {
            t[i][k] = (uint8_t)(xorshift32() >> 26);
        }
    }
    struct SiftMatch m[NQ];
    const int32_t    n = sift_match(m, q, NQ, t, NT, 0.95f);
    int32_t          j = 0;
    for (int32_t i = 0; i < NQ; i++) {
        long long d[NT];
        int       best = 0;
        for (int k = 0; k < NT; k++) {
            d[k] = 0;
            for (int b = 0; b < SIFT_DESC_BYTES; b++) {
                const long long v = (long long)q[i][b] - (long long)t[k][b];
                d[k] += v * v;
            }
            best = d[k] < d[best] ? k : best;
        }
        long long second = -1;
        for (int k = 0; k < NT; k++) {
            if (k != best && (second < 0 || d[k] < second)) {
                second = d[k];
            }
        }
        if ((double)d[best] < 0.9025 * (double)second) {
            CHECK(j < n);
            if (j < n) {
                CHECK_EQ_INT(m[j].query, i);
                CHECK_EQ_INT(m[j].train, best);
                CHECK_EQ_INT(m[j].distance, d[best]);
            }
            j++;
        }
    }
    CHECK_EQ_INT(n, j);
}

// ---- Misuse and scratch -----------------------------------------------------

static void test_misuse(void) {
    struct Mat         *img = texture(40, 40, 1u);
    struct Mat         *rgb = alloc_mat(40, 40, CV_8UC3);
    struct SiftKeyPoint kps[4];
    struct SiftMatch    m[1];
    static uint8_t      d[1][SIFT_DESC_BYTES];
    struct SiftParams   p;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, rgb, NULL));
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, -1, img, NULL));
    EXPECT_ASSERT(sift_detect_compute(NULL, NULL, 4, img, NULL));
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, NULL, NULL));
    sift_default_params(&p);
    p.n_octave_layers = 0;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    p.n_octave_layers = SIFT_MAX_LAYERS + 1;
    EXPECT_ASSERT(sift_scratch_bytes(40, 40, &p));
    sift_default_params(&p);
    p.sigma = 0.0f;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.edge_threshold = 0.0f;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.contrast_threshold = -0.1f;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    EXPECT_ASSERT(sift_scratch_bytes(-1, 40, NULL));
    EXPECT_ASSERT(sift_match(m, d, 1, d, 1, 0.0f));
    EXPECT_ASSERT(sift_match(m, d, 1, d, 1, 1.5f));
    EXPECT_ASSERT(sift_match(m, d, -1, d, 1, 0.8f));
    EXPECT_ASSERT(sift_match(NULL, d, 1, d, 1, 0.8f));
    EXPECT_ASSERT(sift_match(m, d, 1, NULL, 1, 0.8f));
    free_mat(&img);
    free_mat(&rgb);
}

static void test_scratch_peak(void) {
    static struct SiftKeyPoint kps[400];
    static uint8_t             desc[400][SIFT_DESC_BYTES];
    struct Mat                *img  = texture(48, 64, 5u);
    struct Mat                *tiny = texture(2, 40, 6u);
    struct SiftParams          p;
    sift_default_params(&p);
    for (int up = 0; up <= 1; up++) {
        p.upsample = up;
        alloc_counter_start();
        sift_detect_compute(kps, desc, 400, img, &p);
        alloc_counter_stop();
        const size_t want = sift_scratch_bytes(48, 64, &p);
        CHECK(alloc_stats.peak >= want && alloc_stats.peak <= want + 64);
        CHECK_EQ_INT(alloc_stats.allocs, 1);
        CHECK_EQ_INT(alloc_stats.frees, 1);
    }
    alloc_counter_start();
    CHECK_EQ_INT(sift_detect_compute(kps, desc, 400, tiny, NULL), 0);
    alloc_counter_stop();
    CHECK_EQ_INT(alloc_stats.allocs, 0);
    free_mat(&img);
    free_mat(&tiny);
}

static void *failing_alloc(size_t size) {
    (void)size;
    return NULL;
}

static void test_alloc_failure(void) {
    static struct SiftKeyPoint kps[8];
    struct Mat                *img = texture(32, 32, 8u);
    improc_set_allocator(failing_alloc, free);
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 8, img, NULL));
    improc_set_allocator(NULL, NULL);
    free_mat(&img);
}

int main(void) {
    RUN_TEST(test_exp_accuracy);
    RUN_TEST(test_atan2_accuracy);
    RUN_TEST(test_sincos_accuracy);
    RUN_TEST(test_round_floor);
    RUN_TEST(test_reflect101);
    RUN_TEST(test_blur_sweep);
    RUN_TEST(test_octave_count);
    RUN_TEST(test_layer_sigmas);
    RUN_TEST(test_blob);
    RUN_TEST(test_rotate90);
    RUN_TEST(test_size_sweep);
    RUN_TEST(test_capacity_and_null_desc);
    RUN_TEST(test_descriptor_norm);
    RUN_TEST(test_match_hand);
    RUN_TEST(test_match_random);
    RUN_TEST(test_misuse);
    RUN_TEST(test_scratch_peak);
    RUN_TEST(test_alloc_failure);
    return test_finish();
}

#endif // IMPROC_HAS_FLOAT
