#include "sift.h"

#if IMPROC_HAS_FLOAT

#include <math.h> // sqrtf and sqrt only: IEEE-754 rounds them correctly
#include <string.h>

#include "improc_internal.h"
#include "utils.h"

// Every function below uses only +, -, *, /, sqrtf and sqrt, and this file
// is built with -ffp-contract=off, so all targets compute the same bits.
// libm's expf, powf, sinf and cosf differ between C libraries; cv2 calls
// them, so the code below ports cv2's and glibc's versions instead.

#define SIFT_MAX_LAYERS     15   // n_octave_layers limit (fixed-size arrays)
#define SIFT_IMG_BORDER     5    // extrema keep this distance from the edge
#define SIFT_MAX_INTERP     5    // sub-pixel refinement steps
#define SIFT_ORI_BINS       36   // orientation histogram bins
#define SIFT_ORI_SIG_FCTR   1.5f // orientation window sigma / keypoint scale
#define SIFT_ORI_RADIUS     4.5f // 3 * SIFT_ORI_SIG_FCTR
#define SIFT_ORI_PEAK_RATIO 0.8f
#define SIFT_DESCR_WIDTH    4 // d: d x d cells
#define SIFT_DESCR_BINS     8 // n: orientations per cell
#define SIFT_DESCR_SCL_FCTR 3.0f
#define SIFT_DESCR_MAG_THR  0.2f
#define SIFT_DESCR_FCTR     512.0f
#define SIFT_HIST_LEN                                                          \
    ((SIFT_DESCR_WIDTH + 2) * (SIFT_DESCR_WIDTH + 2) * (SIFT_DESCR_BINS + 2))

#define F_EPSILON 1.1920928955078125e-7f // FLT_EPSILON

// ---- Portable float math ----------------------------------------------------

// floor(v) for |v| < 2^31.
static int32_t floor_i(float v) {
    int32_t i = (int32_t)v;
    return (float)i > v ? i - 1 : i;
}

// Nearest integer, ties to even, as cvRound, for |v| < 2^31. v - floor(v) is
// exact in float, so 0.49999997f stays below the tie.
static int32_t round_i(float v) {
    int32_t     i = floor_i(v);
    const float f = v - (float)i;
    if (f > 0.5f || (f == 0.5f && (i & 1) != 0)) {
        i++;
    }
    return i;
}

// 2^k for -126 <= k <= 127, built from the IEEE-754 bits.
static float pow2_i(int32_t k) {
    const uint32_t bits = (uint32_t)(k + 127) << 23;
    float          f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// floor(v) and nearest (ties to even, as cvRound) for |v| < 2^31.
static int32_t floor_id(double v) {
    int32_t i = (int32_t)v;
    return (double)i > v ? i - 1 : i;
}

static int32_t round_id(double v) {
    int32_t      i = floor_id(v);
    const double f = v - (double)i;
    if (f > 0.5 || (f == 0.5 && (i & 1) != 0)) {
        i++;
    }
    return i;
}

// e^x in double for -700 <= x <= 0.7 (a kernel tap or 2^(1/s)), within
// about 2 ulp; 0 below -700.
static double exp_d(double x) {
    if (x < -700.0) {
        return 0.0;
    }
    const int32_t k = round_id(x * 1.4426950408889634);
    // ln 2 = LN2_HI + LN2_LO, LN2_HI with 32 low zero bits: k * LN2_HI exact.
    const double r = (x - (double)k * 6.93147180369123816490e-01) -
                     (double)k * 1.90821492927058770002e-10;
    // Taylor series to r^13 on |r| <= ln(2) / 2.
    double t = 1.0 / 6227020800.0;
    t        = t * r + 1.0 / 479001600.0;
    t        = t * r + 1.0 / 39916800.0;
    t        = t * r + 1.0 / 3628800.0;
    t        = t * r + 1.0 / 362880.0;
    t        = t * r + 1.0 / 40320.0;
    t        = t * r + 1.0 / 5040.0;
    t        = t * r + 1.0 / 720.0;
    t        = t * r + 1.0 / 120.0;
    t        = t * r + 1.0 / 24.0;
    t        = t * r + 1.0 / 6.0;
    t        = t * r + 0.5;
    t        = t * r + 1.0;
    t        = t * r + 1.0;
    // 2^k from the bits; k >= -1010 here, so it is a normal double.
    const uint64_t bits = (uint64_t)(k + 1023) << 52;
    double         p2;
    memcpy(&p2, &bits, sizeof p2);
    return t * p2;
}

// 2^y as glibc powf(2.0f, y), which cv2 calls for the keypoint size: for
// x = 2 its log2 step is exactly 1, which leaves exp2_inline: 2^(k / 32) from
// a table times a cubic, in double. Checked bit for bit against glibc for
// every float y in [-2, 2].
static float pow2_cv(float y) {
    static const uint64_t tab[32] = {
        0x3ff0000000000000u, 0x3fefd9b0d3158574u, 0x3fefb5586cf9890fu,
        0x3fef9301d0125b51u, 0x3fef72b83c7d517bu, 0x3fef54873168b9aau,
        0x3fef387a6e756238u, 0x3fef1e9df51fdee1u, 0x3fef06fe0a31b715u,
        0x3feef1a7373aa9cbu, 0x3feedea64c123422u, 0x3feece086061892du,
        0x3feebfdad5362a27u, 0x3feeb42b569d4f82u, 0x3feeab07dd485429u,
        0x3feea47eb03a5585u, 0x3feea09e667f3bcdu, 0x3fee9f75e8ec5f74u,
        0x3feea11473eb0187u, 0x3feea589994cce13u, 0x3feeace5422aa0dbu,
        0x3feeb737b0cdc5e5u, 0x3feec49182a3f090u, 0x3feed503b23e255du,
        0x3feee89f995ad3adu, 0x3feeff76f2fb5e47u, 0x3fef199bdd85529cu,
        0x3fef3720dcef9069u, 0x3fef5818dcfba487u, 0x3fef7c97337b9b5fu,
        0x3fefa4afa2a490dau, 0x3fefd0765b6e4540u,
    };
    const double shift = 0x1.8p+47; // 1.5 * 2^52 / 32
    const double xd    = (double)y;
    double       kd    = xd + shift; // k / 32 in the low bits
    uint64_t     ki;
    memcpy(&ki, &kd, sizeof ki);
    kd -= shift;
    const double   r = xd - kd;
    const uint64_t t = tab[ki % 32u] + (ki << 47);
    double         sc;
    memcpy(&sc, &t, sizeof sc);
    const double z  = 0x1.c6af84b912394p-5 * r + 0x1.ebfce50fac4f3p-3;
    const double r2 = r * r;
    double       v  = 0x1.62e42ff0c52d6p-1 * r + 1.0;
    v               = z * r2 + v;
    return (float)(v * sc);
}

// e^x as cv::hal::exp32f: 2^(k / 64) from a table times a cubic, all in
// float. Relative error about 1e-7.
static float exp_cv(float x) {
    static const float tab[64] = {
        0x1.3ce0f4p-7f, 0x1.40544ep-7f, 0x1.43d146p-7f, 0x1.4757f6p-7f,
        0x1.4ae87cp-7f, 0x1.4e82f2p-7f, 0x1.522772p-7f, 0x1.55d61ap-7f,
        0x1.598f06p-7f, 0x1.5d5254p-7f, 0x1.61201ep-7f, 0x1.64f882p-7f,
        0x1.68db9ep-7f, 0x1.6cc992p-7f, 0x1.70c278p-7f, 0x1.74c672p-7f,
        0x1.78d59ep-7f, 0x1.7cf01ap-7f, 0x1.811606p-7f, 0x1.854782p-7f,
        0x1.8984bp-7f,  0x1.8dcdaep-7f, 0x1.9222ap-7f,  0x1.9683a4p-7f,
        0x1.9af0dcp-7f, 0x1.9f6a6cp-7f, 0x1.a3f076p-7f, 0x1.a8831cp-7f,
        0x1.ad228p-7f,  0x1.b1cec8p-7f, 0x1.b68816p-7f, 0x1.bb4e9p-7f,
        0x1.c0225ap-7f, 0x1.c50398p-7f, 0x1.c9f272p-7f, 0x1.ceef0cp-7f,
        0x1.d3f98cp-7f, 0x1.d91218p-7f, 0x1.de38dcp-7f, 0x1.e36dfap-7f,
        0x1.e8b19cp-7f, 0x1.ee03ecp-7f, 0x1.f36512p-7f, 0x1.f8d536p-7f,
        0x1.fe5482p-7f, 0x1.01f192p-6f, 0x1.04c0ap-6f,  0x1.079784p-6f,
        0x1.0a7652p-6f, 0x1.0d5d2p-6f,  0x1.104c04p-6f, 0x1.134316p-6f,
        0x1.16426cp-6f, 0x1.194a1ep-6f, 0x1.1c5a42p-6f, 0x1.1f72eep-6f,
        0x1.22943ep-6f, 0x1.25be46p-6f, 0x1.28f122p-6f, 0x1.2c2ce8p-6f,
        0x1.2f71bp-6f,  0x1.32bf96p-6f, 0x1.3616b2p-6f, 0x1.39771ep-6f,
    };
    const float lim = 0x1.03ee22p+11f; // 3000 * 64 / (64 log2 e)
    x               = x < -lim ? -lim : x > lim ? lim : x;
    x *= 0x1.715476p+6f; // 64 log2 e
    const int32_t  k  = round_i(x);
    const float    f  = (x - (float)k) * (1.0f / 64.0f);
    const uint32_t lo = (uint32_t)k & 63u;
    // floor(k / 64) + 127, without a shift of a negative value.
    int32_t e = (k - (int32_t)lo) / 64 + 127;
    e         = e < 0 ? 0 : e > 255 ? 255 : e;
    const uint32_t bits = (uint32_t)e << 23;
    float          p2;
    memcpy(&p2, &bits, sizeof p2);
    return p2 * tab[lo] *
           ((((f + 0x1.6f547ap+2f) * f + 0x1.8d76c8p+4f) * f + 0x1.1eb5acp+6f) *
                f +
            0x1.9da274p+6f);
}

// atan2(y, x) in degrees as cv::hal::fastAtan2: [0, 360], 0 for (0, 0),
// error up to about 0.01 degrees.
static float atan2_deg(float y, float x) {
    const float ax = x < 0.0f ? -x : x;
    const float ay = y < 0.0f ? -y : y;
    float       c, a;
    if (ax >= ay) {
        c = ay / (ax + 0x1p-52f); // + (float)DBL_EPSILON
    } else {
        c = ax / (ay + 0x1p-52f);
    }
    const float c2 = c * c;
    // atan2_p1 .. p7 of cv2, already times 180 / pi.
    a = (((-0x1.4515b2p+1f * c2 + 0x1.1d3f7ep+3f) * c2 - 0x1.2aaddcp+4f) * c2 +
         0x1.ca44dep+5f) *
        c;
    if (ax < ay) {
        a = 90.0f - a;
    }
    if (x < 0.0f) {
        a = 180.0f - a;
    }
    if (y < 0.0f) {
        a = 360.0f - a;
    }
    return a;
}

// sinf and cosf as glibc (ARM optimized-routines), which cv2 calls for the
// descriptor rotation: range reduction to [-pi/4, pi/4], then polynomials in
// double. Checked bit for bit against glibc for every float in [-7, 7];
// SIFT passes only [0, 2 pi]. Above 7 a few values can differ by 1 ulp from
// glibc's FMA variant.
struct sincos_poly {
    double c0, c1, c2, c3, c4, s1, s2, s3;
};

// Polynomials; the second negates the cosine.
static const struct sincos_poly SC_POLY[2] = {
    {0x1p0, -0x1.ffffffd0c621cp-2, 0x1.55553e1068f19p-5,
     -0x1.6c087e89a359dp-10, 0x1.99343027bf8c3p-16, -0x1.555545995a603p-3,
     0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13},
    {-0x1p0, 0x1.ffffffd0c621cp-2, -0x1.55553e1068f19p-5,
     0x1.6c087e89a359dp-10, -0x1.99343027bf8c3p-16, -0x1.555545995a603p-3,
     0x1.1107605230bc4p-7, -0x1.994eb3774cf24p-13},
};

// Sine (n even) or cosine (n odd) polynomial of the reduced x.
static float sc_poly(double x, double x2, const struct sincos_poly *p,
                     int32_t n) {
    if (((uint32_t)n & 1u) == 0) {
        const double x3 = x * x2;
        const double s1 = p->s2 + x2 * p->s3;
        const double x7 = x3 * x2;
        const double sv = x + x3 * p->s1;
        return (float)(sv + x7 * s1);
    }
    const double x4 = x2 * x2;
    const double c2 = p->c3 + x2 * p->c4;
    const double c1 = p->c0 + x2 * p->c1;
    const double x6 = x4 * x2;
    const double cv = c1 + x4 * p->c2;
    return (float)(cv + x6 * c2);
}

// x mod pi/2 for |x| < 120, quadrant in *np.
static double sc_reduce_fast(double x, int32_t *np) {
    const double  r = x * 0x1.45f306dc9c883p+23; // 2 / pi * 2^24
    const int32_t t = (int32_t)r + 0x800000;
    // floor(t / 2^24), without a shift of a negative value.
    const int32_t n = (t - (int32_t)((uint32_t)t & 0xffffffu)) / 16777216;
    *np             = n;
    return x - (double)n * 0x1.921fb54442d18p0;
}

// x mod pi/2 for |x| >= 120 (bits xi, sign ignored), quadrant in *np:
// exact fixed-point product with 192 bits of 4 / pi.
static double sc_reduce_large(uint32_t xi, int32_t *np) {
    static const uint32_t inv_pio4[24] = {
        0xa2u,       0xa2f9u,     0xa2f983u,   0xa2f9836eu, 0xf9836e4eu,
        0x836e4e44u, 0x6e4e4415u, 0x4e441529u, 0x441529fcu, 0x1529fc27u,
        0x29fc2757u, 0xfc2757d1u, 0x2757d1f5u, 0x57d1f534u, 0xd1f534ddu,
        0xf534ddc0u, 0x34ddc0dbu, 0xddc0db62u, 0xc0db6295u, 0xdb629599u,
        0x6295993cu, 0x95993c43u, 0x993c4390u, 0x3c439041u,
    };
    const uint32_t *arr   = &inv_pio4[(xi >> 26) & 15u];
    const uint32_t  shift = (xi >> 23) & 7u;
    xi                    = ((xi & 0xffffffu) | 0x800000u) << shift;
    uint64_t       res0   = (uint32_t)(xi * arr[0]);
    const uint64_t res1   = (uint64_t)xi * arr[4];
    const uint64_t res2   = (uint64_t)xi * arr[8];
    res0                  = (res2 >> 32) | (res0 << 32);
    res0 += res1;
    const uint64_t n = (res0 + ((uint64_t)1 << 61)) >> 62;
    res0 -= n << 62;
    // res0 is a two's complement value in [-2^61, 2^61].
    const double x = (res0 >> 63) != 0 ? -(double)(~res0 + 1u) : (double)res0;
    *np            = (int32_t)n;
    return x * 0x1.921fb54442d18p-62; // 2 pi * 2^-64
}

// sin (cos_ = 0) or cos (cos_ = 1) of a finite y, as glibc.
static float sincos_cv(float y, int32_t cos_) {
    uint32_t bits;
    memcpy(&bits, &y, sizeof bits);
    const uint32_t top = (bits >> 20) & 0x7ffu;
    double         x   = (double)y;
    if (top < 0x3f4u) { // |y| < about pi / 4
        if (top < 0x398u) { // |y| < about 2^-12
            return cos_ ? 1.0f : y;
        }
        return sc_poly(x, x * x, &SC_POLY[0], cos_);
    }
    int32_t n, q;
    if (top < 0x42fu) { // |y| < about 120
        x = sc_reduce_fast(x, &n);
        q = n;
    } else {
        x = sc_reduce_large(bits, &n);
        q = n + (int32_t)(bits >> 31);
    }
    const uint32_t qu = (uint32_t)q & 3u;
    const double   sg = qu == 0u || qu == 3u ? 1.0 : -1.0;
    return sc_poly(x * sg, x * x, &SC_POLY[(qu & 2u) ? 1 : 0],
                   cos_ ? (int32_t)((uint32_t)n ^ 1u) : n);
}

// ---- Scale space ------------------------------------------------------------

// A float image inside the scratch; rows are cols floats apart.
struct fimg {
    float  *d;
    int32_t rows;
    int32_t cols;
};

static inline float px(const struct fimg *m, int32_t r, int32_t c) {
    return m->d[(size_t)r * (size_t)m->cols + (size_t)c];
}

// BORDER_REFLECT_101 index for any i (the kernel may be wider than n).
static int32_t reflect101(int32_t i, int32_t n) {
    if (n == 1) {
        return 0;
    }
    const int32_t period = 2 * n - 2;
    if (i < 0) {
        i = -i;
    }
    i %= period;
    return i < n ? i : period - i;
}

// Gaussian kernel radius for sigma, as cv2 for float images:
// ksize = cvRound(sigma * 8 + 1) | 1, in double.
static int32_t kernel_radius(double sigma) {
    return (round_id(sigma * 4.0 * 2.0 + 1.0) | 1) / 2;
}

// Weights w[0..r] of the half kernel, w[0] at the centre, as
// cv::getGaussianKernel for CV_32F: in double, outer taps first, then
// rounded to float.
static void kernel_weights(float *w, int32_t r, double sigma) {
    const double scale = -0.125 / (sigma * sigma);
    double       sum   = 0.0;
    for (int32_t i = 0; i < r; i++) {
        const int32_t x = 2 * (i - r); // 1 - n + 2i for n = 2r + 1
        sum += exp_d((double)(x * x) * scale);
    }
    const double mul = 1.0 / (sum * 2.0 + 1.0);
    for (int32_t i = 0; i < r; i++) {
        const int32_t x = 2 * (i - r);
        w[r - i]        = (float)(exp_d((double)(x * x) * scale) * mul);
    }
    w[0] = (float)mul;
}

// Row pass, as cv2's RowFilter for kernels wider than 5: the taps kx[0..2r]
// summed left to right (reflect 101 border). The interior runs tap by tap
// over the whole span, which keeps each pixel's order and vectorizes.
static void blur_row(float *restrict dst, const float *restrict src, int32_t n,
                     const float *kx, int32_t r) {
    const int32_t lo = r < n ? r : n;
    const int32_t hi = n - r > lo ? n - r : lo;
    for (int32_t i = 0; i < n; i++) {
        if (i == lo) {
            i = hi;
            if (i >= n) {
                break;
            }
        }
        float sum = kx[0] * src[reflect101(i - r, n)];
        for (int32_t j = 1; j <= 2 * r; j++) {
            sum += kx[j] * src[reflect101(i - r + j, n)];
        }
        dst[i] = sum;
    }
    const int32_t span = hi - lo;
    if (span == 0) {
        return; // no interior; src + (lo - r) may be outside the row
    }
    float *restrict d  = dst + lo;
    const float *sp    = src + (lo - r);
    for (int32_t i = 0; i < span; i++) {
        d[i] = kx[0] * sp[i];
    }
    for (int32_t j = 1; j <= 2 * r; j++) {
        const float  w = kx[j];
        const float *q = sp + j;
        for (int32_t i = 0; i < span; i++) {
            d[i] += w * q[i];
        }
    }
}

// Column pass, as cv2's SymmColumnFilter: centre tap, then c[k] times the
// pair k rows below and above, c[k] = kx[r + k]. Runs a whole row of cols
// pixels at a time.
static void blur_col(float *restrict dst, const float *restrict src,
                     int32_t rows, size_t cols, const float *c, int32_t r) {
    for (int32_t i = 0; i < rows; i++) {
        float *restrict d  = dst + (size_t)i * cols;
        const float    *s0 = src + (size_t)i * cols;
        for (size_t x = 0; x < cols; x++) {
            d[x] = c[0] * s0[x];
        }
        for (int32_t k = 1; k <= r; k++) {
            const float *a = src + (size_t)reflect101(i + k, rows) * cols;
            const float *b = src + (size_t)reflect101(i - k, rows) * cols;
            const float  w = c[k];
            for (size_t x = 0; x < cols; x++) {
                d[x] += w * (a[x] + b[x]);
            }
        }
    }
}

// dst = Gaussian(src, sigma); tmp has the same size. All three distinct. kx
// holds 2r + 1 floats.
static void gauss_blur(struct fimg *dst, const struct fimg *src,
                       struct fimg *tmp, double sigma, float *kx) {
    const int32_t r = kernel_radius(sigma);
    kernel_weights(kx, r, sigma);
    // Half kernel w[0..r] to the full kernel, centre at kx[r]. Descending k
    // reads w[k] before kx[r + k] overwrites anything not yet read.
    for (int32_t k = r; k >= 0; k--) {
        kx[r + k] = kx[k];
    }
    for (int32_t k = 1; k <= r; k++) {
        kx[r - k] = kx[r + k];
    }
    const size_t cols = (size_t)src->cols;
    for (int32_t y = 0; y < src->rows; y++) {
        blur_row(tmp->d + (size_t)y * cols, src->d + (size_t)y * cols,
                 src->cols, kx, r);
    }
    blur_col(dst->d, tmp->d, src->rows, cols, kx + r, r);
    dst->rows = src->rows;
    dst->cols = src->cols;
}

// Octave count, as cv2: round(log2(min(R, C))) - 2 for octave 0 of R x C.
static int32_t octave_count(int32_t r0, int32_t c0) {
    const uint64_t m = (uint64_t)(r0 < c0 ? r0 : c0);
    int32_t        l = 0; // largest l with m * m >= 2^(2l - 1)
    while (l < 31 && m * m >= ((uint64_t)1 << (2 * l + 1))) {
        l++;
    }
    return l - 2;
}

// Blur from layer i - 1 to layer i, in double as cv2: k = 2^(1/s),
// prev = k^(i-1) sigma, sig[i] = sqrt((prev k)^2 - prev^2).
static void layer_sigmas(double *sig, const struct SiftParams *p) {
    const int32_t s = p->n_octave_layers;
    const double  k = exp_d(0.69314718055994531 / (double)s);
    double        kp = 1.0; // k^(i-1)
    sig[0]           = p->sigma;
    for (int32_t i = 1; i < s + 3; i++) {
        const double prev  = kp * p->sigma;
        const double total = prev * k;
        sig[i]             = sqrt(total * total - prev * prev);
        kp *= k;
    }
}

// Base blur in float as cv2, from (float)sigma. The input is assumed blurred
// by 0.5 (1.0 after the 2x upsample).
static float base_sigma(const struct SiftParams *p) {
    const float sg   = (float)p->sigma;
    const float have = p->upsample ? 1.0f : 0.25f;
    float       v    = sg * sg - have;
    return sqrtf(v > 0.01f ? v : 0.01f);
}

static void check_params(const struct SiftParams *p) {
    ASSERT(p->n_octave_layers >= 1 && p->n_octave_layers <= SIFT_MAX_LAYERS);
    // NaN fails every comparison.
    ASSERT(p->contrast_threshold >= 0.0f && p->contrast_threshold <= 100.0f);
    ASSERT(p->edge_threshold > 0.0f && p->edge_threshold <= 1.0e4f);
    ASSERT(p->sigma >= 0.1 && p->sigma <= 32.0);
}

void sift_default_params(struct SiftParams *p) {
    ASSERT(p != NULL);
    p->n_octave_layers    = 3;
    p->contrast_threshold = 0.04f;
    p->edge_threshold     = 10.0f;
    p->sigma              = 1.6;
    p->upsample           = 1;
}

// Octave 0 size; 0 x 0 when there is no octave.
static void octave0_size(int rows, int cols, const struct SiftParams *p,
                         int32_t *r0, int32_t *c0) {
    ASSERT(!p->upsample || (rows <= INT32_MAX / 2 && cols <= INT32_MAX / 2));
    *r0 = p->upsample ? 2 * (int32_t)rows : (int32_t)rows;
    *c0 = p->upsample ? 2 * (int32_t)cols : (int32_t)cols;
    if (octave_count(*r0, *c0) < 1) {
        *r0 = *c0 = 0;
    }
}

// Largest kernel radius of any blur: the base blur or a layer step.
static int32_t max_radius(const struct SiftParams *p) {
    double sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, p);
    int32_t r = kernel_radius((double)base_sigma(p));
    for (int32_t i = 1; i < p->n_octave_layers + 3; i++) {
        const int32_t ri = kernel_radius(sig[i]);
        r                = ri > r ? ri : r;
    }
    return r;
}

size_t sift_scratch_bytes(int rows, int cols, const struct SiftParams *p) {
    struct SiftParams def;
    if (p == NULL) {
        sift_default_params(&def);
        p = &def;
    }
    check_params(p);
    ASSERT(rows >= 0 && cols >= 0);
    int32_t r0, c0;
    octave0_size(rows, cols, p, &r0, &c0);
    if (r0 == 0) {
        return 0;
    }
    // n_octave_layers + 3 Gaussian images and one blur buffer, then the
    // descriptor histogram and the blur kernel.
    const size_t img = improc_scratch_mul((size_t)r0, (size_t)c0);
    const size_t images =
        improc_scratch_mul(img, (size_t)(p->n_octave_layers + 4));
    const size_t extra = improc_scratch_add(
        (size_t)SIFT_HIST_LEN,
        improc_scratch_add(improc_scratch_mul(2, (size_t)max_radius(p)), 1));
    return improc_scratch_mul(improc_scratch_add(images, extra), sizeof(float));
}

// ---- Keypoint store: min-heap on (response, detection order) ----------------

struct store {
    struct SiftKeyPoint *kps;
    uint8_t (*desc)[SIFT_DESC_BYTES];
    int32_t capacity;
    int32_t n;
};

// 1 when a should be evicted before b: lower response, or the same response
// and later in the fixed order (octave, layer, y, x, angle).
static int weaker(const struct SiftKeyPoint *a, const struct SiftKeyPoint *b) {
    if (a->response != b->response) {
        return a->response < b->response;
    }
    if (a->octave != b->octave) {
        return a->octave > b->octave;
    }
    if (a->layer != b->layer) {
        return a->layer > b->layer;
    }
    if (a->y != b->y) {
        return a->y > b->y;
    }
    if (a->x != b->x) {
        return a->x > b->x;
    }
    return a->angle > b->angle;
}

static void store_swap(struct store *st, int32_t i, int32_t j) {
    const struct SiftKeyPoint t = st->kps[i];
    st->kps[i]                  = st->kps[j];
    st->kps[j]                  = t;
    if (st->desc != NULL) {
        uint8_t td[SIFT_DESC_BYTES];
        memcpy(td, st->desc[i], SIFT_DESC_BYTES);
        memcpy(st->desc[i], st->desc[j], SIFT_DESC_BYTES);
        memcpy(st->desc[j], td, SIFT_DESC_BYTES);
    }
}

static void heap_down(struct store *st, int32_t i, int32_t n) {
    for (;;) {
        const int32_t l = 2 * i + 1;
        if (l >= n) {
            return;
        }
        int32_t m = l;
        if (l + 1 < n && weaker(&st->kps[l + 1], &st->kps[l])) {
            m = l + 1;
        }
        if (!weaker(&st->kps[m], &st->kps[i])) {
            return;
        }
        store_swap(st, i, m);
        i = m;
    }
}

static void heap_up(struct store *st, int32_t i) {
    while (i > 0) {
        const int32_t parent = (i - 1) / 2;
        if (!weaker(&st->kps[i], &st->kps[parent])) {
            return;
        }
        store_swap(st, i, parent);
        i = parent;
    }
}

// Slot for a new keypoint kp, or -1 when the store is full and kp is weaker
// than all kept ones. The caller fills the slot and calls store_commit.
static int32_t store_slot(const struct store        *st,
                          const struct SiftKeyPoint *kp) {
    if (st->n < st->capacity) {
        return st->n;
    }
    return weaker(&st->kps[0], kp) ? 0 : -1;
}

static void store_commit(struct store *st, int32_t slot) {
    if (slot == st->n) {
        st->n++;
        heap_up(st, slot);
    } else {
        heap_down(st, 0, st->n);
    }
}

// Heapsort: the min-heap becomes an array in decreasing strength.
static void store_sort(struct store *st) {
    for (int32_t end = st->n - 1; end > 0; end--) {
        store_swap(st, 0, end);
        heap_down(st, 0, end);
    }
}

// ---- Detection --------------------------------------------------------------

struct octave {
    struct fimg g[SIFT_MAX_LAYERS + 3]; // Gaussian images of this octave
    int32_t     index;                  // 0 = octave 0 (upsampled or not)
};

static inline float dog(const struct octave *o, int32_t l, int32_t r,
                        int32_t c) {
    return px(&o->g[l + 1], r, c) - px(&o->g[l], r, c);
}

// Solves H x = b (3 x 3) by Cramer's rule in float, as cv2
// Matx33f::solve(DECOMP_LU). A singular H gives x = 0, as cv2.
static void solve3(const float h[3][3], const float b[3], float x[3]) {
    float d = h[0][0] * (h[1][1] * h[2][2] - h[2][1] * h[1][2]) -
              h[0][1] * (h[1][0] * h[2][2] - h[2][0] * h[1][2]) +
              h[0][2] * (h[1][0] * h[2][1] - h[2][0] * h[1][1]);
    if (d == 0.0f) {
        x[0] = x[1] = x[2] = 0.0f;
        return;
    }
    d    = 1.0f / d;
    x[0] = d * (b[0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
                h[0][1] * (b[1] * h[2][2] - h[1][2] * b[2]) +
                h[0][2] * (b[1] * h[2][1] - h[1][1] * b[2]));
    x[1] = d * (h[0][0] * (b[1] * h[2][2] - h[1][2] * b[2]) -
                b[0] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                h[0][2] * (h[1][0] * b[2] - b[1] * h[2][0]));
    x[2] = d * (h[0][0] * (h[1][1] * b[2] - b[1] * h[2][1]) -
                h[0][1] * (h[1][0] * b[2] - b[1] * h[2][0]) +
                b[0] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]));
}

// Gradient (dD) and Hessian (H) of the DoG at (layer, r, c), image values
// scaled to 0..1.
static void dog_derivs(const struct octave *o, int32_t l, int32_t r, int32_t c,
                       float d[3], float h[3][3]) {
    const float img_scale = 1.0f / 255.0f;
    const float deriv     = img_scale * 0.5f;
    const float cross     = img_scale * 0.25f;
    const float v2        = dog(o, l, r, c) * 2.0f;
    d[0]                  = (dog(o, l, r, c + 1) - dog(o, l, r, c - 1)) * deriv;
    d[1]                  = (dog(o, l, r + 1, c) - dog(o, l, r - 1, c)) * deriv;
    d[2]                  = (dog(o, l + 1, r, c) - dog(o, l - 1, r, c)) * deriv;
    const float dxx =
        (dog(o, l, r, c + 1) + dog(o, l, r, c - 1) - v2) * img_scale;
    const float dyy =
        (dog(o, l, r + 1, c) + dog(o, l, r - 1, c) - v2) * img_scale;
    const float dss =
        (dog(o, l + 1, r, c) + dog(o, l - 1, r, c) - v2) * img_scale;
    const float dxy = (dog(o, l, r + 1, c + 1) - dog(o, l, r + 1, c - 1) -
                       dog(o, l, r - 1, c + 1) + dog(o, l, r - 1, c - 1)) *
                      cross;
    const float dxs = (dog(o, l + 1, r, c + 1) - dog(o, l + 1, r, c - 1) -
                       dog(o, l - 1, r, c + 1) + dog(o, l - 1, r, c - 1)) *
                      cross;
    const float dys = (dog(o, l + 1, r + 1, c) - dog(o, l + 1, r - 1, c) -
                       dog(o, l - 1, r + 1, c) + dog(o, l - 1, r - 1, c)) *
                      cross;
    h[0][0]         = dxx;
    h[0][1]         = dxy;
    h[0][2]         = dxs;
    h[1][0]         = dxy;
    h[1][1]         = dyy;
    h[1][2]         = dys;
    h[2][0]         = dxs;
    h[2][1]         = dys;
    h[2][2]         = dss;
}

// A refined extremum in octave coordinates.
struct extremum {
    int32_t r, c, layer; // integer position after refinement
    float   xr, xc, xi;  // sub-pixel offsets, each in (-0.5, 0.5)
    float   contrast;    // DoG value at the refined position (0..1 scale)
};

// Refines the extremum at (layer, r, c) as cv2 adjustLocalExtrema. Returns 0
// when it moves out of range, does not converge, has low contrast or lies on
// an edge.
static int refine(const struct octave *o, const struct SiftParams *p,
                  int32_t layer, int32_t r, int32_t c, struct extremum *e) {
    const int32_t rows = o->g[0].rows, cols = o->g[0].cols;
    const int32_t s = p->n_octave_layers;
    float         d[3], h[3][3], x[3] = {0.0f, 0.0f, 0.0f};
    int32_t       i;
    for (i = 0; i < SIFT_MAX_INTERP; i++) {
        dog_derivs(o, layer, r, c, d, h);
        solve3(h, d, x);
        const float xc = -x[0], xr = -x[1], xi = -x[2];
        if (xc < 0.5f && xc > -0.5f && xr < 0.5f && xr > -0.5f && xi < 0.5f &&
            xi > -0.5f) {
            break;
        }
        const float big = 715827882.0f; // (float)(INT_MAX / 3), as cv2
        if (xc > big || xc < -big || xr > big || xr < -big || xi > big ||
            xi < -big) {
            return 0;
        }
        c += round_i(xc);
        r += round_i(xr);
        layer += round_i(xi);
        if (layer < 1 || layer > s || c < SIFT_IMG_BORDER ||
            c >= cols - SIFT_IMG_BORDER || r < SIFT_IMG_BORDER ||
            r >= rows - SIFT_IMG_BORDER) {
            return 0;
        }
    }
    if (i >= SIFT_MAX_INTERP) {
        return 0;
    }
    e->xc = -x[0];
    e->xr = -x[1];
    e->xi = -x[2];
    dog_derivs(o, layer, r, c, d, h);
    const float t  = d[0] * e->xc + d[1] * e->xr + d[2] * e->xi;
    e->contrast    = dog(o, layer, r, c) * (1.0f / 255.0f) + t * 0.5f;
    const float ac = e->contrast < 0.0f ? -e->contrast : e->contrast;
    if (ac * (float)s < p->contrast_threshold) {
        return 0;
    }
    const float tr  = h[0][0] + h[1][1];
    const float det = h[0][0] * h[1][1] - h[0][1] * h[0][1];
    const float et  = p->edge_threshold;
    if (det <= 0.0f || tr * tr * et >= (et + 1.0f) * (et + 1.0f) * det) {
        return 0;
    }
    e->r     = r;
    e->c     = c;
    e->layer = layer;
    return 1;
}

// Gradient of img at (r, c): dx to the right, dy upwards (as cv2).
static void grad(const struct fimg *img, int32_t r, int32_t c, float *dx,
                 float *dy) {
    *dx = px(img, r, c + 1) - px(img, r, c - 1);
    *dy = px(img, r - 1, c) - px(img, r + 1, c);
}

// Smoothed 36-bin orientation histogram around (r, c); returns its maximum.
static float orientation_hist(const struct fimg *img, int32_t r, int32_t c,
                              int32_t radius, float sigma, float *hist) {
    float         raw[SIFT_ORI_BINS + 4];
    float        *t     = raw + 2;
    const float   scale = -1.0f / (2.0f * sigma * sigma);
    const int32_t n     = SIFT_ORI_BINS;
    for (int32_t i = 0; i < n; i++) {
        t[i] = 0.0f;
    }
    for (int32_t i = -radius; i <= radius; i++) {
        const int32_t y = r + i;
        if (y <= 0 || y >= img->rows - 1) {
            continue;
        }
        for (int32_t j = -radius; j <= radius; j++) {
            const int32_t x = c + j;
            if (x <= 0 || x >= img->cols - 1) {
                continue;
            }
            float dx, dy;
            grad(img, y, x, &dx, &dy);
            const float w   = exp_cv((float)(i * i + j * j) * scale);
            const float mag = sqrtf(dx * dx + dy * dy);
            int32_t     bin = round_i((float)n / 360.0f * atan2_deg(dy, dx));
            if (bin >= n) {
                bin -= n;
            }
            t[bin] += w * mag;
        }
    }
    t[-1]      = t[n - 1];
    t[-2]      = t[n - 2];
    t[n]       = t[0];
    t[n + 1]   = t[1];
    float best = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        // cv2 order: a + (b + c), its SIMD loop covers all 36 bins.
        hist[i] = (t[i - 2] + t[i + 2]) * (1.0f / 16.0f) +
                  ((t[i - 1] + t[i + 1]) * (4.0f / 16.0f) +
                   t[i] * (6.0f / 16.0f));
        best    = hist[i] > best ? hist[i] : best;
    }
    return best;
}

// 128-byte descriptor of a keypoint at (x, y) in img, orientation ori
// degrees (y up, as the gradient), scale scl; cv2 calcSIFTDescriptor.
static void describe(uint8_t *out, const struct fimg *img, float x, float y,
                     float ori, float scl, float *hist) {
    const int32_t d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_BINS;
    const int32_t pr = round_i(y), pc = round_i(x);
    const float   rad = ori * 0x1.1df46ap-6f; // (float)(CV_PI / 180)
    float         cos_t = sincos_cv(rad, 1), sin_t = sincos_cv(rad, 0);
    const float bins_per_deg = (float)n / 360.0f;
    const float exp_scale    = -1.0f / ((float)(d * d) * 0.5f);
    const float hist_width   = SIFT_DESCR_SCL_FCTR * scl;
    int32_t radius = round_i(hist_width * 1.4142135623730951f * (float)(d + 1) *
                             0.5f);
    const int32_t diag = (int32_t)sqrt((double)img->cols * (double)img->cols +
                                       (double)img->rows * (double)img->rows);
    radius             = radius < diag ? radius : diag;
    cos_t /= hist_width;
    sin_t /= hist_width;
    for (int32_t i = 0; i < SIFT_HIST_LEN; i++) {
        hist[i] = 0.0f;
    }
    for (int32_t i = -radius; i <= radius; i++) {
        for (int32_t j = -radius; j <= radius; j++) {
            const float   c_rot = (float)j * cos_t - (float)i * sin_t;
            const float   r_rot = (float)j * sin_t + (float)i * cos_t;
            float         rbin  = r_rot + (float)(d / 2) - 0.5f;
            float         cbin  = c_rot + (float)(d / 2) - 0.5f;
            const int32_t r = pr + i, c = pc + j;
            if (!(rbin > -1.0f && rbin < (float)d && cbin > -1.0f &&
                  cbin < (float)d && r > 0 && r < img->rows - 1 && c > 0 &&
                  c < img->cols - 1)) {
                continue;
            }
            float dx, dy;
            grad(img, r, c, &dx, &dy);
            const float w =
                exp_cv((c_rot * c_rot + r_rot * r_rot) * exp_scale);
            const float mag  = sqrtf(dx * dx + dy * dy) * w;
            float       obin = (atan2_deg(dy, dx) - ori) * bins_per_deg;
            const int32_t r0 = floor_i(rbin), c0 = floor_i(cbin);
            int32_t       o0 = floor_i(obin);
            rbin -= (float)r0;
            cbin -= (float)c0;
            obin -= (float)o0;
            if (o0 < 0) {
                o0 += n;
            }
            if (o0 >= n) {
                o0 -= n;
            }
            // Trilinear split over the 2 x 2 x 2 neighbouring bins.
            const float   v_r1 = mag * rbin, v_r0 = mag - v_r1;
            const float   v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
            const float   v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
            const float   v111 = v_rc11 * obin, v110 = v_rc11 - v111;
            const float   v101 = v_rc10 * obin, v100 = v_rc10 - v101;
            const float   v011 = v_rc01 * obin, v010 = v_rc01 - v011;
            const float   v001 = v_rc00 * obin, v000 = v_rc00 - v001;
            const int32_t idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
            hist[idx] += v000;
            hist[idx + 1] += v001;
            hist[idx + (n + 2)] += v010;
            hist[idx + (n + 3)] += v011;
            hist[idx + (d + 2) * (n + 2)] += v100;
            hist[idx + (d + 2) * (n + 2) + 1] += v101;
            hist[idx + (d + 3) * (n + 2)] += v110;
            hist[idx + (d + 3) * (n + 2) + 1] += v111;
        }
    }
    // Wrap the orientation bins, then gather the d x d x n values.
    // The first norm is cv2's 4-lane SIMD sum: lane k % 4, then
    // (l0 + l2) + (l1 + l3).
    float   v[SIFT_DESC_BYTES];
    float   lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t k       = 0;
    for (int32_t i = 0; i < d; i++) {
        for (int32_t j = 0; j < d; j++) {
            const int32_t idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];
            for (int32_t o = 0; o < n; o++) {
                v[k] = hist[idx + o];
                lane[(uint32_t)k & 3u] += v[k] * v[k];
                k++;
            }
        }
    }
    float       nrm2 = (lane[0] + lane[2]) + (lane[1] + lane[3]);
    const float thr  = sqrtf(nrm2) * SIFT_DESCR_MAG_THR;
    nrm2             = 0.0f;
    for (k = 0; k < SIFT_DESC_BYTES; k++) {
        v[k] = v[k] < thr ? v[k] : thr;
        nrm2 += v[k] * v[k];
    }
    const float norm = sqrtf(nrm2);
    const float f    = SIFT_DESCR_FCTR / (norm > F_EPSILON ? norm : F_EPSILON);
    for (k = 0; k < SIFT_DESC_BYTES; k++) {
        const int32_t q = round_i(v[k] * f);
        out[k]          = (uint8_t)(q > 255 ? 255 : q);
    }
}

struct ctx {
    const struct SiftParams   *p;
    struct store               st;
    float                     *hist; // descriptor histogram, SIFT_HIST_LEN
    struct fimg               *tmp;  // blur buffer, free between blurs
    const struct SiftKeyPoint *in;   // sift_compute: the given keypoints
    uint8_t (*in_desc)[SIFT_DESC_BYTES];
    int32_t n_in;
};

// Descriptor of a finished keypoint, unpacked from its public fields as cv2
// calcDescriptors does, so detect_compute and compute give the same bytes.
static void describe_kp(uint8_t *out, const struct octave *o,
                        const struct SiftKeyPoint *kp, float *hist) {
    const float scale = kp->octave >= 0 ? 1.0f / pow2_i(kp->octave) : 2.0f;
    // The descriptor turns the other way: y up, as the gradients.
    float ori = 360.0f - kp->angle;
    if (ori - 360.0f < F_EPSILON && ori - 360.0f > -F_EPSILON) {
        ori = 0.0f;
    }
    describe(out, &o->g[kp->layer], kp->x * scale, kp->y * scale, ori,
             kp->size * scale * 0.5f, hist);
}

// Orientation peaks of a refined extremum; one keypoint per peak.
static void add_keypoints(struct ctx *cx, const struct octave *o,
                          const struct extremum *e) {
    const struct SiftParams *p = cx->p;
    const float              layer_scale =
        pow2_cv(((float)e->layer + e->xi) / (float)p->n_octave_layers);
    const float   scl = (float)p->sigma * layer_scale; // octave pixels
    const float   oct_scale = pow2_i(o->index) * (p->upsample ? 0.5f : 1.0f);
    float         hist[SIFT_ORI_BINS];
    const float   omax = orientation_hist(&o->g[e->layer], e->r, e->c,
                                          round_i(SIFT_ORI_RADIUS * scl),
                                          SIFT_ORI_SIG_FCTR * scl, hist);
    const float   thr  = omax * SIFT_ORI_PEAK_RATIO;
    const int32_t n    = SIFT_ORI_BINS;
    for (int32_t j = 0; j < n; j++) {
        const int32_t l = j > 0 ? j - 1 : n - 1;
        const int32_t r = j < n - 1 ? j + 1 : 0;
        if (!(hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= thr)) {
            continue;
        }
        float bin   = (float)j + 0.5f * (hist[l] - hist[r]) /
                                     (hist[l] - 2.0f * hist[j] + hist[r]);
        bin         = bin < 0.0f        ? (float)n + bin
                      : bin >= (float)n ? bin - (float)n
                                        : bin;
        float angle = 360.0f - 360.0f / (float)n * bin;
        if (angle >= 360.0f) {
            angle = 0.0f;
        }
        struct SiftKeyPoint kp;
        kp.x               = ((float)e->c + e->xc) * oct_scale;
        kp.y               = ((float)e->r + e->xr) * oct_scale;
        kp.size            = scl * 2.0f * oct_scale;
        kp.angle           = angle;
        kp.response        = e->contrast < 0.0f ? -e->contrast : e->contrast;
        kp.octave          = o->index - (p->upsample ? 1 : 0);
        kp.layer           = e->layer;
        const int32_t slot = store_slot(&cx->st, &kp);
        if (slot < 0) {
            continue;
        }
        cx->st.kps[slot] = kp;
        if (cx->st.desc != NULL) {
            describe_kp(cx->st.desc[slot], o, &kp, cx->hist);
        }
        store_commit(&cx->st, slot);
    }
}

_Static_assert(sizeof(uint32_t) == sizeof(float), "seen map in blur buffer");

// Scans the DoG of one octave for extrema and stores their keypoints.
static void find_extrema(struct ctx *cx, const struct octave *o) {
    const struct SiftParams *p    = cx->p;
    const int32_t            s    = p->n_octave_layers;
    const int32_t            rows = o->g[0].rows, cols = o->g[0].cols;
    // Two start pixels can refine to the same extremum and so give the same
    // keypoints; cv2 drops such copies before it keeps the strongest. Bit
    // l - 1 of seen[r * cols + c] marks a stored extremum (the blur buffer
    // is free until the next octave).
    uint32_t *seen = (uint32_t *)(void *)cx->tmp->d;
    memset(seen, 0, (size_t)rows * (size_t)cols * sizeof(uint32_t));
    // cv2: floor(0.5 * contrast / s * 255) in double.
    const float thr = (float)floor_id(0.5 * (double)p->contrast_threshold /
                                      (double)s * 255.0);
    for (int32_t l = 1; l <= s; l++) {
        for (int32_t r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++) {
            for (int32_t c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++) {
                const float v  = dog(o, l, r, c);
                const float av = v < 0.0f ? -v : v;
                if (!(av > thr)) {
                    continue;
                }
                int ext = 1;
                for (int32_t dl = -1; dl <= 1 && ext; dl++) {
                    for (int32_t dr = -1; dr <= 1 && ext; dr++) {
                        for (int32_t dc = -1; dc <= 1; dc++) {
                            const float u = dog(o, l + dl, r + dr, c + dc);
                            if (v > 0.0f ? u > v : u < v) {
                                ext = 0;
                                break;
                            }
                        }
                    }
                }
                struct extremum e;
                if (!ext || !refine(o, p, l, r, c, &e)) {
                    continue;
                }
                uint32_t *m = &seen[(size_t)e.r * (size_t)cols + (size_t)e.c];
                const uint32_t bit = (uint32_t)1 << (e.layer - 1);
                if ((*m & bit) == 0) {
                    *m |= bit;
                    add_keypoints(cx, o, &e);
                }
            }
        }
    }
}

// sift_compute: the descriptors of the given keypoints in this octave.
static void describe_given(struct ctx *cx, const struct octave *o) {
    const int32_t num = o->index - (cx->p->upsample ? 1 : 0);
    for (int32_t i = 0; i < cx->n_in; i++) {
        if (cx->in[i].octave == num) {
            describe_kp(cx->in_desc[i], o, &cx->in[i], cx->hist);
        }
    }
}

// Octave 0, layer 0: the input (2x bilinear when upsampling) blurred to
// sigma. tmp and up are scratch images of octave 0 size.
static void build_base(struct fimg *g0, struct fimg *up, struct fimg *tmp,
                       const struct Mat *src, const struct SiftParams *p,
                       float *w) {
    const int32_t rows = src->rows, cols = src->cols;
    if (!p->upsample) {
        for (int32_t y = 0; y < rows; y++) {
            const uchar *s = mat_row(src, y);
            float       *d = up->d + (size_t)y * (size_t)cols;
            for (int32_t x = 0; x < cols; x++) {
                d[x] = (float)s[x];
            }
        }
    } else {
        // cv2.resize INTER_LINEAR x2: output pixel 2i samples source
        // i - 0.25, pixel 2i + 1 samples i + 0.25; indices clamp at the edge.
        const size_t uc = 2 * (size_t)cols;
        for (int32_t y = 0; y < 2 * rows; y++) {
            const int32_t y0 = y / 2 - ((y & 1) ? 0 : 1);
            const float   fy = (y & 1) ? 0.25f : 0.75f;
            const uchar  *s0 = mat_row(src, y0 < 0 ? 0 : y0);
            const uchar  *s1 = mat_row(src, y0 + 1 >= rows ? rows - 1 : y0 + 1);
            float        *d  = up->d + (size_t)y * uc;
            for (int32_t x = 0; x < 2 * cols; x++) {
                int32_t     x0 = x / 2 - ((x & 1) ? 0 : 1);
                int32_t     x1 = x0 + 1;
                const float fx = (x & 1) ? 0.25f : 0.75f;
                x0             = x0 < 0 ? 0 : x0;
                x1             = x1 >= cols ? cols - 1 : x1;
                const float top =
                    (float)s0[x0] * (1.0f - fx) + (float)s0[x1] * fx;
                const float bot =
                    (float)s1[x0] * (1.0f - fx) + (float)s1[x1] * fx;
                d[x] = top * (1.0f - fy) + bot * fy;
            }
        }
    }
    up->rows  = p->upsample ? 2 * rows : rows;
    up->cols  = p->upsample ? 2 * cols : cols;
    tmp->rows = up->rows;
    tmp->cols = up->cols;
    gauss_blur(g0, up, tmp, (double)base_sigma(p), w);
}

// Builds octaves 0 .. octaves - 1 of the scale space one at a time with the
// parameters cx->p and hands each to visit.
static void scale_space(struct ctx *cx, const struct Mat *src, int32_t octaves,
                        void (*visit)(struct ctx *, const struct octave *)) {
    const struct SiftParams *p = cx->p;
    int32_t                  r0, c0;
    octave0_size(src->rows, src->cols, p, &r0, &c0);
    const size_t bytes     = sift_scratch_bytes(src->rows, src->cols, p);
    float       *buf       = (float *)fast_malloc(bytes);
    void        *blocks[1] = {buf};
    improc_scratch_check(blocks, 1);

    const int32_t s      = p->n_octave_layers;
    const size_t  img    = (size_t)r0 * (size_t)c0;
    float        *weight = buf + img * (size_t)(s + 4) + SIFT_HIST_LEN;
    struct octave o;
    struct fimg   tmp;
    for (int32_t i = 0; i < s + 3; i++) {
        o.g[i].d = buf + img * (size_t)i;
    }
    tmp.d    = buf + img * (size_t)(s + 3);
    cx->hist = buf + img * (size_t)(s + 4);
    cx->tmp  = &tmp;

    double sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, p);
    // Layer 1 holds the unblurred base while layer 0 is built from it.
    build_base(&o.g[0], &o.g[1], &tmp, src, p, weight);
    for (o.index = 0; o.index < octaves; o.index++) {
        if (o.index > 0) {
            // Every second pixel of layer s of the octave before.
            const struct fimg *prev = &o.g[s];
            const int32_t      nr = prev->rows / 2, nc = prev->cols / 2;
            for (int32_t y = 0; y < nr; y++) {
                for (int32_t x = 0; x < nc; x++) {
                    o.g[0].d[(size_t)y * (size_t)nc + (size_t)x] =
                        px(prev, 2 * y, 2 * x);
                }
            }
            o.g[0].rows = nr;
            o.g[0].cols = nc;
        }
        tmp.rows = o.g[0].rows;
        tmp.cols = o.g[0].cols;
        for (int32_t i = 1; i < s + 3; i++) {
            gauss_blur(&o.g[i], &o.g[i - 1], &tmp, sig[i], weight);
        }
        visit(cx, &o);
    }
    fast_free(buf);
}

static const struct SiftParams *params_or_default(const struct SiftParams *p,
                                                  struct SiftParams *def) {
    if (p == NULL) {
        sift_default_params(def);
        p = def;
    }
    check_params(p);
    return p;
}

static void check_src(const struct Mat *src) {
    ASSERT(src != NULL && mat_type(src) == CV_8UC1 && src->data != NULL);
}

int32_t sift_detect_compute(struct SiftKeyPoint *kps,
                            uint8_t (*desc)[SIFT_DESC_BYTES], int32_t capacity,
                            const struct Mat *src, const struct SiftParams *p) {
    struct SiftParams def;
    p = params_or_default(p, &def);
    check_src(src);
    ASSERT(capacity >= 0 && (capacity == 0 || kps != NULL));
    if (capacity == 0) {
        return 0;
    }
    int32_t r0, c0;
    octave0_size(src->rows, src->cols, p, &r0, &c0);
    if (r0 == 0) {
        return 0;
    }
    struct ctx cx;
    memset(&cx, 0, sizeof cx);
    cx.p           = p;
    cx.st.kps      = kps;
    cx.st.desc     = desc;
    cx.st.capacity = capacity;
    scale_space(&cx, src, octave_count(r0, c0), find_extrema);
    store_sort(&cx.st);
    return cx.st.n;
}

int32_t sift_detect(struct SiftKeyPoint *kps, int32_t capacity,
                    const struct Mat *src, const struct SiftParams *p) {
    return sift_detect_compute(kps, NULL, capacity, src, p);
}

void sift_compute(uint8_t (*desc)[SIFT_DESC_BYTES],
                  const struct SiftKeyPoint *kps, int32_t n,
                  const struct Mat *src, const struct SiftParams *p) {
    struct SiftParams def;
    p = params_or_default(p, &def);
    check_src(src);
    ASSERT(n >= 0 && (n == 0 || (kps != NULL && desc != NULL)));
    // As cv2: upsample exactly when some keypoint is in octave -1.
    struct SiftParams q   = *p;
    int32_t           top = 0;
    q.upsample            = 0;
    for (int32_t i = 0; i < n; i++) {
        const struct SiftKeyPoint *k = &kps[i];
        ASSERT(k->octave >= -1 && k->octave <= 30);
        ASSERT(k->layer >= 0 && k->layer <= p->n_octave_layers + 2);
        ASSERT(k->x >= -1.0e6f && k->x <= 1.0e6f && k->y >= -1.0e6f &&
               k->y <= 1.0e6f);
        ASSERT(k->size > 0.0f && k->size <= 1.0e6f);
        ASSERT(k->angle >= 0.0f && k->angle <= 360.0f);
        q.upsample = q.upsample || k->octave < 0;
        top        = k->octave > top ? k->octave : top;
    }
    if (n == 0) {
        return;
    }
    int32_t r0, c0;
    octave0_size(src->rows, src->cols, &q, &r0, &c0);
    const int32_t octaves = top + (q.upsample ? 2 : 1);
    ASSERT(r0 > 0 && octaves <= octave_count(r0, c0));
    struct ctx cx;
    memset(&cx, 0, sizeof cx);
    cx.p       = &q;
    cx.in      = kps;
    cx.in_desc = desc;
    cx.n_in    = n;
    scale_space(&cx, src, octaves, describe_given);
}
// ---- Matching ---------------------------------------------------------------

static int32_t l2sq(const uint8_t *a, const uint8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < SIFT_DESC_BYTES; i++) {
        const int32_t d = (int32_t)a[i] - (int32_t)b[i];
        sum += d * d;
    }
    return sum;
}

int32_t sift_match(struct SiftMatch *matches,
                   const uint8_t (*query)[SIFT_DESC_BYTES], int32_t nq,
                   const uint8_t (*train)[SIFT_DESC_BYTES], int32_t nt,
                   float ratio) {
    ASSERT(nq >= 0 && nt >= 0);
    ASSERT(ratio > 0.0f && ratio <= 1.0f);
    ASSERT(nq == 0 || (matches != NULL && query != NULL));
    ASSERT(nt == 0 || train != NULL);
    const float r2  = ratio * ratio;
    int32_t     out = 0;
    if (nt == 0) {
        return 0;
    }
    for (int32_t q = 0; q < nq; q++) {
        int32_t best = -1, d1 = INT32_MAX, d2 = INT32_MAX;
        for (int32_t t = 0; t < nt; t++) {
            const int32_t d = l2sq(query[q], train[t]);
            if (d < d1) {
                d2   = d1;
                d1   = d;
                best = t;
            } else if (d < d2) {
                d2 = d;
            }
        }
        // d1 < r2 * d2, both exact in float (d <= 128 * 255^2 < 2^24).
        if (nt == 1 || (float)d1 < r2 * (float)d2) {
            matches[out].query    = q;
            matches[out].train    = best;
            matches[out].distance = d1;
            out++;
        }
    }
    return out;
}

#endif // IMPROC_HAS_FLOAT
