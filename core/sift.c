#include "sift.h"

#if IMPROC_HAS_FLOAT

#include <math.h> // sqrtf only: IEEE-754 requires it to be correctly rounded
#include <string.h>

#include "improc_internal.h"
#include "utils.h"

// Every float function below uses only +, -, *, / and sqrtf, and this file is
// built with -ffp-contract=off, so all targets compute the same bits. libm's
// expf, atan2f, sinf and cosf differ between C libraries and are not used.

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

#define F_PI      3.14159265358979f
#define F_LN2     0.693147180559945f
#define F_LOG2E   1.44269504088896f
#define F_RAD2DEG 57.2957795130823f
#define F_DEG2RAD 0.0174532925199433f
#define F_EPSILON 1.1920928955078125e-7f // FLT_EPSILON

// ---- Portable float math ----------------------------------------------------

// floor(v) for |v| < 2^31.
static int32_t floor_i(float v) {
    int32_t i = (int32_t)v;
    return (float)i > v ? i - 1 : i;
}

static int32_t round_i(float v) { return floor_i(v + 0.5f); }

// 2^k for -126 <= k <= 127, built from the IEEE-754 bits.
static float pow2_i(int32_t k) {
    const uint32_t bits = (uint32_t)(k + 127) << 23;
    float          f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// e^x. Relative error below 2e-7 for -87 <= x <= 88; 0 below -87.
static float exp_f(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    if (x > 88.0f) {
        x = 88.0f;
    }
    const int32_t k = round_i(x * F_LOG2E);
    // ln 2 split in two so k * LN2_HI is exact for |k| < 2^9.
    const float r = (x - (float)k * 0.693145751953125f) -
                    (float)k * 1.42860682030941723212e-6f;
    // Taylor series to r^7 on |r| <= ln(2) / 2.
    float p = 1.0f / 5040.0f;
    p       = p * r + 1.0f / 720.0f;
    p       = p * r + 1.0f / 120.0f;
    p       = p * r + 1.0f / 24.0f;
    p       = p * r + 1.0f / 6.0f;
    p       = p * r + 0.5f;
    p       = p * r + 1.0f;
    p       = p * r + 1.0f;
    return p * pow2_i(k);
}

// atan(t) for 0 <= t <= 1, Abramowitz & Stegun 4.4.49 (error <= 2e-8).
static float atan_unit(float t) {
    const float t2 = t * t;
    float       p  = 0.0028662257f;
    p              = p * t2 - 0.0161657367f;
    p              = p * t2 + 0.0429096138f;
    p              = p * t2 - 0.0752896400f;
    p              = p * t2 + 0.1065626393f;
    p              = p * t2 - 0.1420889944f;
    p              = p * t2 + 0.1999355085f;
    p              = p * t2 - 0.3333314528f;
    return t + t * t2 * p;
}

// atan2(y, x) in degrees, [0, 360); 0 for (0, 0). As cv2.fastAtan2, but
// accurate to about 1e-5 degrees.
static float atan2_deg(float y, float x) {
    const float ax = x < 0.0f ? -x : x;
    const float ay = y < 0.0f ? -y : y;
    if (ax == 0.0f && ay == 0.0f) {
        return 0.0f;
    }
    float a = ay <= ax ? atan_unit(ay / ax) : F_PI * 0.5f - atan_unit(ax / ay);
    if (x < 0.0f) {
        a = F_PI - a;
    }
    float d = a * F_RAD2DEG;
    if (y < 0.0f) {
        d = 360.0f - d;
    }
    return d >= 360.0f ? d - 360.0f : d;
}

// sin and cos of deg degrees, 0 <= deg < 720.
static void sincos_deg(float deg, float *s, float *c) {
    // Quadrant q and remainder r in [-45, 45] degrees.
    const int32_t q  = round_i(deg * (1.0f / 90.0f));
    const float   r  = (deg - (float)q * 90.0f) * F_DEG2RAD;
    const float   r2 = r * r;
    // Taylor series: |r| <= pi / 4, errors below 4e-8.
    float sp       = -1.0f / 39916800.0f;
    sp             = sp * r2 + 1.0f / 362880.0f;
    sp             = sp * r2 - 1.0f / 5040.0f;
    sp             = sp * r2 + 1.0f / 120.0f;
    sp             = sp * r2 - 1.0f / 6.0f;
    const float sr = r + r * r2 * sp;
    float       cp = 1.0f / 3628800.0f;
    cp             = cp * r2 - 1.0f / 40320.0f;
    cp             = cp * r2 + 1.0f / 720.0f;
    cp             = cp * r2 - 1.0f / 24.0f;
    cp             = cp * r2 + 0.5f;
    const float cr = 1.0f - r2 * cp;
    switch (q & 3) {
    case 0:
        *s = sr;
        *c = cr;
        break;
    case 1:
        *s = cr;
        *c = -sr;
        break;
    case 2:
        *s = -sr;
        *c = -cr;
        break;
    default:
        *s = -cr;
        *c = sr;
        break;
    }
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
// ksize = round(sigma * 8 + 1) | 1.
static int32_t kernel_radius(float sigma) {
    return (round_i(sigma * 8.0f + 1.0f) | 1) / 2;
}

// Normalised weights w[0..r] of the half kernel, w[0] at the centre.
static void kernel_weights(float *w, int32_t r, float sigma) {
    const float scale = -0.5f / (sigma * sigma);
    float       sum   = 0.0f;
    for (int32_t k = 0; k <= r; k++) {
        w[k] = exp_f((float)(k * k) * scale);
        sum += k == 0 ? w[k] : 2.0f * w[k];
    }
    const float inv = 1.0f / sum;
    for (int32_t k = 0; k <= r; k++) {
        w[k] *= inv;
    }
}

// One 1-D pass along a line of n values with stride st (reflect 101 border).
static void blur_line(float *restrict dst, const float *restrict src, int32_t n,
                      size_t st, const float *w, int32_t r) {
    for (int32_t i = 0; i < n; i++) {
        float sum = w[0] * src[(size_t)i * st];
        if (i >= r && i + r < n) {
            for (int32_t k = 1; k <= r; k++) {
                sum += w[k] *
                       (src[(size_t)(i - k) * st] + src[(size_t)(i + k) * st]);
            }
        } else {
            for (int32_t k = 1; k <= r; k++) {
                sum += w[k] * (src[(size_t)reflect101(i - k, n) * st] +
                               src[(size_t)reflect101(i + k, n) * st]);
            }
        }
        dst[(size_t)i * st] = sum;
    }
}

// dst = Gaussian(src, sigma); tmp has the same size. All three distinct.
static void gauss_blur(struct fimg *dst, const struct fimg *src,
                       struct fimg *tmp, float sigma, float *w) {
    const int32_t r = kernel_radius(sigma);
    kernel_weights(w, r, sigma);
    const size_t cols = (size_t)src->cols;
    for (int32_t y = 0; y < src->rows; y++) {
        blur_line(tmp->d + (size_t)y * cols, src->d + (size_t)y * cols,
                  src->cols, 1, w, r);
    }
    for (int32_t x = 0; x < src->cols; x++) {
        blur_line(dst->d + x, tmp->d + x, src->rows, cols, w, r);
    }
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

static void layer_sigmas(float *sig, const struct SiftParams *p) {
    const int32_t s = p->n_octave_layers;
    const float   k = exp_f(F_LN2 / (float)s); // 2^(1/s)
    sig[0]          = p->sigma;
    float prev      = p->sigma;
    for (int32_t i = 1; i < s + 3; i++) {
        const float total = prev * k;
        sig[i]            = sqrtf(total * total - prev * prev);
        prev              = total;
    }
}

static float base_sigma(const struct SiftParams *p) {
    // The input is assumed blurred by 0.5 (1.0 after the 2x upsample).
    const float have = p->upsample ? 1.0f : 0.25f;
    float       v    = p->sigma * p->sigma - have;
    return sqrtf(v > 0.01f ? v : 0.01f);
}

static void check_params(const struct SiftParams *p) {
    ASSERT(p->n_octave_layers >= 1 && p->n_octave_layers <= SIFT_MAX_LAYERS);
    ASSERT(p->contrast_threshold >= 0.0f && p->edge_threshold > 0.0f);
    ASSERT(p->sigma > 0.0f && p->sigma <= 32.0f);
}

void sift_default_params(struct SiftParams *p) {
    ASSERT(p != NULL);
    p->n_octave_layers    = 3;
    p->contrast_threshold = 0.04f;
    p->edge_threshold     = 10.0f;
    p->sigma              = 1.6f;
    p->upsample           = 1;
}

// Octave 0 size; 0 x 0 when there is no octave.
static void octave0_size(int rows, int cols, const struct SiftParams *p,
                         int32_t *r0, int32_t *c0) {
    *r0 = p->upsample ? 2 * (int32_t)rows : (int32_t)rows;
    *c0 = p->upsample ? 2 * (int32_t)cols : (int32_t)cols;
    if (octave_count(*r0, *c0) < 1) {
        *r0 = *c0 = 0;
    }
}

// Largest kernel radius of any blur: the base blur or a layer step.
static int32_t max_radius(const struct SiftParams *p) {
    float sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, p);
    int32_t r = kernel_radius(base_sigma(p));
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
    // descriptor histogram and the half kernel.
    const size_t img = improc_scratch_mul((size_t)r0, (size_t)c0);
    const size_t images =
        improc_scratch_mul(img, (size_t)(p->n_octave_layers + 4));
    const size_t extra = (size_t)SIFT_HIST_LEN + (size_t)max_radius(p) + 1;
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

// Solves H x = b (3 x 3) by Gaussian elimination with partial pivoting.
// Returns 0 when H is singular.
static int solve3(float h[3][3], float b[3], float x[3]) {
    for (int i = 0; i < 3; i++) {
        int   piv = i;
        float big = h[i][i] < 0.0f ? -h[i][i] : h[i][i];
        for (int k = i + 1; k < 3; k++) {
            const float v = h[k][i] < 0.0f ? -h[k][i] : h[k][i];
            if (v > big) {
                big = v;
                piv = k;
            }
        }
        if (big == 0.0f) {
            return 0;
        }
        if (piv != i) {
            for (int j = 0; j < 3; j++) {
                const float t = h[i][j];
                h[i][j]       = h[piv][j];
                h[piv][j]     = t;
            }
            const float t = b[i];
            b[i]          = b[piv];
            b[piv]        = t;
        }
        for (int k = i + 1; k < 3; k++) {
            const float f = h[k][i] / h[i][i];
            for (int j = i; j < 3; j++) {
                h[k][j] -= f * h[i][j];
            }
            b[k] -= f * b[i];
        }
    }
    for (int i = 2; i >= 0; i--) {
        float v = b[i];
        for (int j = i + 1; j < 3; j++) {
            v -= h[i][j] * x[j];
        }
        x[i] = v / h[i][i];
    }
    return 1;
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
        if (!solve3(h, d, x)) {
            return 0;
        }
        const float xc = -x[0], xr = -x[1], xi = -x[2];
        if (xc < 0.5f && xc > -0.5f && xr < 0.5f && xr > -0.5f && xi < 0.5f &&
            xi > -0.5f) {
            break;
        }
        if (xc > 1e6f || xc < -1e6f || xr > 1e6f || xr < -1e6f || xi > 1e6f ||
            xi < -1e6f) {
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
            const float w   = exp_f((float)(i * i + j * j) * scale);
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
        hist[i] = (t[i - 2] + t[i + 2]) * (1.0f / 16.0f) +
                  (t[i - 1] + t[i + 1]) * (4.0f / 16.0f) +
                  t[i] * (6.0f / 16.0f);
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
    float         cos_t, sin_t;
    sincos_deg(ori, &sin_t, &cos_t);
    const float bins_per_deg = (float)n / 360.0f;
    const float exp_scale    = -1.0f / ((float)(d * d) * 0.5f);
    const float hist_width   = SIFT_DESCR_SCL_FCTR * scl;
    int32_t radius = round_i(hist_width * 1.41421356f * (float)(d + 1) * 0.5f);
    const float diag = sqrtf((float)img->rows * (float)img->rows +
                             (float)img->cols * (float)img->cols);
    if ((float)radius > diag) {
        radius = (int32_t)diag;
    }
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
            const float w = exp_f((c_rot * c_rot + r_rot * r_rot) * exp_scale);
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
    float   v[SIFT_DESC_BYTES];
    float   nrm2 = 0.0f;
    int32_t k    = 0;
    for (int32_t i = 0; i < d; i++) {
        for (int32_t j = 0; j < d; j++) {
            const int32_t idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];
            for (int32_t o = 0; o < n; o++) {
                v[k] = hist[idx + o];
                nrm2 += v[k] * v[k];
                k++;
            }
        }
    }
    const float thr = sqrtf(nrm2) * SIFT_DESCR_MAG_THR;
    nrm2            = 0.0f;
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
    const struct SiftParams *p;
    struct store             st;
    float                   *hist;      // descriptor histogram, SIFT_HIST_LEN
    float                    out_scale; // octave 0 pixel -> input pixel
};

// Orientation peaks of a refined extremum; one keypoint per peak.
static void add_keypoints(struct ctx *cx, const struct octave *o,
                          const struct extremum *e) {
    const struct SiftParams *p = cx->p;
    const float              layer_scale =
        exp_f(((float)e->layer + e->xi) / (float)p->n_octave_layers * F_LN2);
    const float   scl       = p->sigma * layer_scale; // octave pixels
    const float   oct_scale = pow2_i(o->index) * cx->out_scale;
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
            // The descriptor turns the other way: y up, as the gradients.
            const float ori = angle == 0.0f ? 0.0f : 360.0f - angle;
            describe(cx->st.desc[slot], &o->g[e->layer], (float)e->c + e->xc,
                     (float)e->r + e->xr, ori, scl, cx->hist);
        }
        store_commit(&cx->st, slot);
    }
}

// Scans the DoG of one octave for extrema and stores their keypoints.
static void find_extrema(struct ctx *cx, const struct octave *o) {
    const struct SiftParams *p    = cx->p;
    const int32_t            s    = p->n_octave_layers;
    const int32_t            rows = o->g[0].rows, cols = o->g[0].cols;
    // cv2: floor(0.5 * contrast / s * 255).
    const float thr =
        (float)floor_i(0.5f * p->contrast_threshold / (float)s * 255.0f);
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
                if (ext && refine(o, p, l, r, c, &e)) {
                    add_keypoints(cx, o, &e);
                }
            }
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
    gauss_blur(g0, up, tmp, base_sigma(p), w);
}

int32_t sift_detect_compute(struct SiftKeyPoint *kps,
                            uint8_t (*desc)[SIFT_DESC_BYTES], int32_t capacity,
                            const struct Mat *src, const struct SiftParams *p) {
    struct SiftParams def;
    if (p == NULL) {
        sift_default_params(&def);
        p = &def;
    }
    check_params(p);
    ASSERT(src != NULL && mat_type(src) == CV_8UC1);
    ASSERT(capacity >= 0 && (capacity == 0 || kps != NULL));
    if (capacity == 0) {
        return 0;
    }
    int32_t r0, c0;
    octave0_size(src->rows, src->cols, p, &r0, &c0);
    if (r0 == 0) {
        return 0;
    }
    const size_t bytes     = sift_scratch_bytes(src->rows, src->cols, p);
    float       *buf       = (float *)fast_malloc(bytes);
    void        *blocks[1] = {buf};
    improc_scratch_check(blocks, 1);

    const int32_t s      = p->n_octave_layers;
    const size_t  img    = (size_t)r0 * (size_t)c0;
    float        *hist   = buf + img * (size_t)(s + 4);
    float        *weight = hist + SIFT_HIST_LEN;
    struct octave o;
    struct fimg   tmp;
    for (int32_t i = 0; i < s + 3; i++) {
        o.g[i].d = buf + img * (size_t)i;
    }
    tmp.d = buf + img * (size_t)(s + 3);

    struct ctx cx;
    cx.p           = p;
    cx.st.kps      = kps;
    cx.st.desc     = desc;
    cx.st.capacity = capacity;
    cx.st.n        = 0;
    cx.hist        = hist;
    cx.out_scale   = p->upsample ? 0.5f : 1.0f;

    float sig[SIFT_MAX_LAYERS + 3];
    layer_sigmas(sig, p);
    // Layer 1 holds the unblurred base while layer 0 is built from it.
    build_base(&o.g[0], &o.g[1], &tmp, src, p, weight);

    const int32_t octaves = octave_count(r0, c0);
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
        find_extrema(&cx, &o);
    }
    fast_free(buf);
    store_sort(&cx.st);
    // Two start pixels can refine to the same extremum: drop the copies, as
    // cv2. Copies compare equal in weaker(), so the sort made them adjacent.
    int32_t n = 0;
    for (int32_t i = 0; i < cx.st.n; i++) {
        const struct SiftKeyPoint *a = &kps[i];
        if (n > 0) {
            const struct SiftKeyPoint *b = &kps[n - 1];
            if (a->x == b->x && a->y == b->y && a->size == b->size &&
                a->angle == b->angle) {
                continue;
            }
        }
        if (n != i) {
            kps[n] = *a;
            if (desc != NULL) {
                memcpy(desc[n], desc[i], SIFT_DESC_BYTES);
            }
        }
        n++;
    }
    return n;
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
