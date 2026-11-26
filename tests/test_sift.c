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

// The private helpers (exp_cv, atan2_deg, gauss_blur, ...) are tested
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

// pow2_cv is glibc powf(2, y), which is not correctly rounded: the
// 0x1.c671d6p-16 case rounds the other way from 2^y.
static void test_pow2_cv(void) {
    static const float y[]  = {0.0f,          0.5f,          1.0f / 3.0f,
                               0.7654321f,    1.1666666f,    0x1.c671d6p-16f,
                               1.999f};
    static const float ref[] = {1.0f,          0x1.6a09e6p+0f, 0x1.428a3p+0f,
                                0x1.b32b46p+0f, 0x1.1f59acp+1f, 0x1.00013cp+0f,
                                0x1.ffa52cp+1f};
    for (int i = 0; i < 7; i++) {
        CHECK(pow2_cv(y[i]) == ref[i]);
    }
    for (int32_t i = 0; i <= 20000; i++) {
        const float  v = (float)i * 0.0001f;
        const double e = exp2((double)v);
        CHECK(fabs((double)pow2_cv(v) - e) <= 1.2e-7 * e);
    }
}

// solve3 is cv2's Cramer solve; a singular H gives 0, as cv2.
static void test_solve3(void) {
    const float h[3][3] = {{2.0f, 1.0f, 0.0f}, {1.0f, 3.0f, 1.0f},
                           {0.0f, 1.0f, 4.0f}};
    const float b[3]    = {3.0f, 5.0f, 5.0f}; // x = (1, 1, 1)
    float       x[3];
    solve3(h, b, x);
    CHECK(x[0] == 1.0f && x[1] == 1.0f && x[2] == 1.0f);
    const float hs[3][3] = {{1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 6.0f},
                            {0.0f, 1.0f, 1.0f}};
    x[0] = x[1] = x[2] = 7.0f;
    solve3(hs, b, x);
    CHECK(x[0] == 0.0f && x[1] == 0.0f && x[2] == 0.0f);
}

// atan2_deg is cv2.fastAtan2: about 0.01 degrees, and the bits of
// cv2.phase(x, y, angleInDegrees=True) (setUseOptimized(False)).
static void test_atan2_accuracy(void) {
    double worst = 0.0;
    for (int iy = -200; iy <= 200; iy++) {
        for (int ix = -200; ix <= 200; ix++) {
            if (ix == 0 && iy == 0) {
                continue;
            }
            const float y = (float)iy * 0.37f, x = (float)ix * 0.37f;
            const float d = atan2_deg(y, x);
            CHECK(d >= 0.0f && d <= 360.0f);
            double ref = atan2((double)y, (double)x) * 180.0 / 3.14159265358979;
            if (ref < 0.0) {
                ref += 360.0;
            }
            double err = fabs((double)d - ref);
            err        = err > 180.0 ? 360.0 - err : err;
            worst      = err > worst ? err : worst;
        }
    }
    CHECK(worst < 0.0096); // cv2.phase on this grid: 0.009552
    static const float y[] = {4.0f, 7.0f, -1.0f, -1e-30f, 1.0f, -3.5f, 0.0f};
    static const float x[] = {3.0f, -2.5f, 0.0f, 1.0f, 1.0f, -3.5f, 0.0f};
    static const float cv[] = {0x1.a9171cp+5f, 0x1.b6a1c4p+6f, 0x1.0ep+8f,
                               0x1.68p+8f,     0x1.67ec74p+5f, 0x1.c1fb1cp+7f,
                               0.0f};
    for (int i = 0; i < 7; i++) {
        CHECK(atan2_deg(y[i], x[i]) == cv[i]);
    }
}

// exp_cv is cv::hal::exp32f: x * 64 log2 e is rounded in float, so the
// relative error grows to about 1e-7 |x|. Also its bits.
static void test_exp_cv(void) {
    for (int32_t i = -40000; i <= 0; i++) {
        const float  x   = (float)i * 0.001f;
        const double ref = exp((double)x);
        CHECK(fabs((double)exp_cv(x) - ref) <=
              (2e-7 + 1e-7 * fabs((double)x)) * ref);
    }
    static const float x[]  = {-0.5f, -3.25f, -17.0f, -0.001f, -39.75f, 0.0f};
    static const float cv[] = {0x1.368b3p-1f,   0x1.3da368p-5f, 0x1.639e2ep-25f,
                               0x1.ff7cfep-1f,  0x1.928214p-58f, 1.0f};
    for (int i = 0; i < 6; i++) {
        CHECK(exp_cv(x[i]) == cv[i]);
    }
    CHECK(exp_cv(-1e30f) == 0.0f);
}

// sincos_cv is glibc sinf / cosf: within 1 ulp of sin and cos, and the
// glibc bits on the small, fast and large reduction paths.
static void test_sincos_cv(void) {
    for (int32_t i = -8000; i <= 8000; i++) {
        const float y = (float)i * 0.001f;
        CHECK(fabs((double)sincos_cv(y, 0) - sin((double)y)) <= 1.2e-7);
        CHECK(fabs((double)sincos_cv(y, 1) - cos((double)y)) <= 1.2e-7);
    }
    static const float y[] = {0.0f,       1e-5f,      0.5f,    0.7853982f,
                              3.0f,       6.2657323f, -5.0f,   119.9f,
                              1000.0f,    -1e20f};
    static const float gs[] = {0.0f,           0x1.4f8b58p-17f, 0x1.eaee88p-2f,
                               0x1.6a09e6p-1f, 0x1.210386p-3f,  -0x1.1def8cp-6f,
                               0x1.eaf82p-1f,  0x1.fc57fep-2f,  0x1.a75cc2p-1f,
                               -0x1.502ad2p-1f};
    static const float gc[] = {1.0f,           1.0f,           0x1.c1528p-1f,
                               0x1.6a09e6p-1f, -0x1.fae04cp-1f, 0x1.ffec0ap-1f,
                               0x1.227858p-2f, 0x1.bc749ap-1f,  0x1.1ff026p-1f,
                               0x1.822e46p-1f};
    for (int i = 0; i < 10; i++) {
        CHECK(sincos_cv(y[i], 0) == gs[i]);
        CHECK(sincos_cv(y[i], 1) == gc[i]);
    }
}

static void test_round_floor(void) {
    CHECK_EQ_INT(floor_i(2.5f), 2);
    CHECK_EQ_INT(floor_i(-2.5f), -3);
    CHECK_EQ_INT(floor_i(-3.0f), -3);
    // Half to even, as cvRound; 0.49999997f + 0.5f is 1.0f in float.
    CHECK_EQ_INT(round_i(2.5f), 2);
    CHECK_EQ_INT(round_i(-2.5f), -2);
    CHECK_EQ_INT(round_i(3.5f), 4);
    CHECK_EQ_INT(round_i(-3.5f), -4);
    CHECK_EQ_INT(round_i(0.49999997f), 0);
    CHECK_EQ_INT(round_i(-0.49999997f), 0);
    CHECK_EQ_INT(round_i(-2.6f), -3);
    CHECK_EQ_INT(round_i(8388609.0f), 8388609);
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
    double sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, &p);
    // Layer i must reach total blur sigma * 2^(i / s).
    double total = p.sigma;
    for (int i = 1; i < p.n_octave_layers + 3; i++) {
        total            = sqrt(total * total + sig[i] * sig[i]);
        const double ref = p.sigma * pow(2.0, (double)i / 3.0);
        CHECK(fabs(total - ref) < 1e-12 * ref);
    }
    // cv2 SIFT sig[1..5] (std::pow and std::sqrt in double), to 2 ulp.
    static const double cv[5] = {1.2262734984654078, 1.5450077936447955,
                                 1.9465878414647133, 2.4525469969308156,
                                 3.090015587289591};
    for (int i = 0; i < 5; i++) {
        CHECK(fabs(sig[i + 1] - cv[i]) <= 4.5e-16 * cv[i]);
    }
}

static void test_exp_d(void) {
    for (int32_t i = -70000; i <= 70; i++) {
        const double x = (double)i * 0.01 + 0.003;
        const double e = exp(x);
        CHECK(fabs(exp_d(x) - e) <= 3.4e-16 * e);
    }
    CHECK(exp_d(-800.0) == 0.0);
    CHECK(exp_d(0.0) == 1.0);
}

// The half kernels of the default layer blurs are the bits of
// cv2.getGaussianKernel(ksize, sig[i], CV_32F), centre first.
static void test_kernel_cv2(void) {
    static const float k11[] = {0x1.4d2364p-2f, 0x1.ddcdfcp-3f, 0x1.606d0ep-4f,
                                0x1.0b5ddep-6f, 0x1.a1407ap-10f,
                                0x1.4edfacp-14f};
    static const float k13[] = {
        0x1.086a76p-2f, 0x1.ace486p-3f, 0x1.c995cap-4f,  0x1.411cd4p-5f,
        0x1.287054p-7f, 0x1.68008p-10f, 0x1.1f90cap-13f};
    static const float k27[] = {
        0x1.0869f6p-3f,  0x1.f5d90ep-4f, 0x1.ace3b6p-4f, 0x1.4a176ap-4f,
        0x1.c994eep-5f,  0x1.1d9ebap-5f, 0x1.411c3ap-6f, 0x1.451d18p-7f,
        0x1.286fc4p-8f,  0x1.e6d384p-10f, 0x1.67ffd2p-11f, 0x1.df7c2p-13f,
        0x1.1f903ep-14f, 0x1.36a04cp-16f};
    struct SiftParams p;
    sift_default_params(&p);
    double sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, &p);
    const struct {
        int          layer;
        const float *ref;
        int32_t      r;
    } cases[] = {{1, k11, 5}, {2, k13, 6}, {5, k27, 13}};
    float w[32];
    for (int c = 0; c < 3; c++) {
        const double sg = sig[cases[c].layer];
        CHECK_EQ_INT(kernel_radius(sg), cases[c].r);
        kernel_weights(w, cases[c].r, sg);
        for (int32_t k = 0; k <= cases[c].r; k++) {
            CHECK(w[k] == cases[c].ref[k]);
        }
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
                CHECK_EQ_INT(bytes, (7 * r0 * c0 + 360 + 27) * sizeof(float));
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
    // Copies are dropped before the cut, so every capacity k gives exactly
    // the first k keypoints of the full result.
    for (int32_t k = 1; k < n; k += k < 40 ? 1 : 7) {
        CHECK_EQ_INT(sift_detect_compute(part, dp, k, img, NULL), k);
        CHECK(memcmp(part, full, sizeof(full[0]) * (size_t)k) == 0);
        CHECK(memcmp(dp, df, SIFT_DESC_BYTES * (size_t)k) == 0);
    }
    CHECK_EQ_INT(sift_detect_compute(nodesc, NULL, CAP, img, NULL), n);
    CHECK(memcmp(nodesc, full, sizeof(full[0]) * (size_t)n) == 0);
    CHECK_EQ_INT(sift_detect_compute(NULL, NULL, 0, img, NULL), 0);
    free_mat(&img);
}

// This texture has two start pixels that refine to the same keypoint, with
// the copy among the strongest 9: the copy must not use up a slot.
static void test_capacity_copies(void) {
    enum { CAP = 200 };
    static struct SiftKeyPoint full[CAP], part[CAP];
    struct Mat                *img = texture(64, 80, 81u);
    const int32_t              n   = sift_detect_compute(full, NULL, CAP, img, NULL);
    CHECK(n > 9 && n < CAP);
    for (int32_t k = 1; k <= n; k++) {
        CHECK_EQ_INT(sift_detect_compute(part, NULL, k, img, NULL), k);
        CHECK(memcmp(part, full, sizeof(full[0]) * (size_t)k) == 0);
    }
    free_mat(&img);
}

// sift_detect then sift_compute gives the bytes of sift_detect_compute, also
// for a keypoint list in another order; without any octave -1 keypoint,
// compute uses the image without the 2x upsample, as cv2.
static void test_detect_then_compute(void) {
    enum { CAP = 600 };
    static struct SiftKeyPoint kd[CAP], kc[CAP], rev[CAP];
    static uint8_t             dd[CAP][SIFT_DESC_BYTES], dc[CAP][SIFT_DESC_BYTES];
    struct Mat                *img = texture(90, 70, 31u);
    const int32_t              n   = sift_detect_compute(kd, dd, CAP, img, NULL);
    CHECK(n > 20 && n < CAP);
    CHECK_EQ_INT(sift_detect(kc, CAP, img, NULL), n);
    CHECK(memcmp(kc, kd, sizeof(kd[0]) * (size_t)n) == 0);
    for (int32_t i = 0; i < n; i++) {
        rev[i] = kd[n - 1 - i];
    }
    sift_compute(dc, rev, n, img, NULL);
    int32_t up = 0;
    for (int32_t i = 0; i < n; i++) {
        CHECK(memcmp(dc[i], dd[n - 1 - i], SIFT_DESC_BYTES) == 0);
        up += kd[i].octave < 0;
    }
    CHECK(up > 0);
    // Octave >= 0 only: same as a detection without upsample.
    struct SiftParams p;
    sift_default_params(&p);
    p.upsample       = 0;
    const int32_t m  = sift_detect_compute(kd, dd, CAP, img, &p);
    CHECK(m > 5);
    sift_compute(dc, kd, m, img, NULL);
    CHECK(memcmp(dc, dd, SIFT_DESC_BYTES * (size_t)m) == 0);
    sift_compute(dc, kd, 0, img, NULL);
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
    p.sigma = 0.0;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.edge_threshold = 0.0f;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.contrast_threshold = -0.1f;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    p.contrast_threshold = INFINITY;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    p.contrast_threshold = NAN;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.edge_threshold = INFINITY;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    sift_default_params(&p);
    p.sigma = 1e-30;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, img, &p));
    p.sigma = NAN;
    EXPECT_ASSERT(sift_scratch_bytes(40, 40, &p));
    EXPECT_ASSERT(sift_scratch_bytes(INT32_MAX / 2 + 1, 40, NULL));
    struct SiftKeyPoint bad = {10.0f, 10.0f, 3.0f, 0.0f, 0.1f, 0, 1};
    uint8_t             bd[1][SIFT_DESC_BYTES];
    EXPECT_ASSERT(sift_compute(bd, &bad, -1, img, NULL));
    EXPECT_ASSERT(sift_compute(NULL, &bad, 1, img, NULL));
    bad.octave = -2;
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    bad.octave = 9; // beyond the octaves of a 40 x 40 image
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    bad.octave = 0;
    bad.layer  = 6;
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    bad.layer = 1;
    bad.size  = NAN;
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    bad.size  = 3.0f;
    bad.angle = INFINITY;
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    bad.angle = 0.0f;
    bad.x     = NAN;
    EXPECT_ASSERT(sift_compute(bd, &bad, 1, img, NULL));
    struct Mat hollow = *img;
    hollow.data       = NULL;
    EXPECT_ASSERT(sift_detect_compute(kps, NULL, 4, &hollow, NULL));
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
    RUN_TEST(test_pow2_cv);
    RUN_TEST(test_solve3);
    RUN_TEST(test_atan2_accuracy);
    RUN_TEST(test_sincos_cv);
    RUN_TEST(test_round_floor);
    RUN_TEST(test_reflect101);
    RUN_TEST(test_blur_sweep);
    RUN_TEST(test_octave_count);
    RUN_TEST(test_layer_sigmas);
    RUN_TEST(test_exp_d);
    RUN_TEST(test_exp_cv);
    RUN_TEST(test_kernel_cv2);
    RUN_TEST(test_blob);
    RUN_TEST(test_rotate90);
    RUN_TEST(test_size_sweep);
    RUN_TEST(test_capacity_and_null_desc);
    RUN_TEST(test_capacity_copies);
    RUN_TEST(test_detect_then_compute);
    RUN_TEST(test_descriptor_norm);
    RUN_TEST(test_match_hand);
    RUN_TEST(test_match_random);
    RUN_TEST(test_misuse);
    RUN_TEST(test_scratch_peak);
    RUN_TEST(test_alloc_failure);
    return test_finish();
}

#endif // IMPROC_HAS_FLOAT
