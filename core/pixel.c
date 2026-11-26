#include "pixel.h"

static void thresh_binary_row(const uchar *restrict s, uchar *restrict d, int n,
                              uchar thresh, uchar maxval) {
    for (int i = 0; i < n; i++) {
        d[i] = s[i] > thresh ? maxval : 0;
    }
}

static void thresh_binary_inv_row(const uchar *restrict s, uchar *restrict d,
                                  int n, uchar thresh, uchar maxval) {
    for (int i = 0; i < n; i++) {
        d[i] = s[i] > thresh ? 0 : maxval;
    }
}

static void thresh_trunc_row(const uchar *restrict s, uchar *restrict d, int n,
                             uchar thresh, uchar maxval) {
    (void)maxval;
    for (int i = 0; i < n; i++) {
        d[i] = s[i] > thresh ? thresh : s[i];
    }
}

static void thresh_tozero_row(const uchar *restrict s, uchar *restrict d, int n,
                              uchar thresh, uchar maxval) {
    (void)maxval;
    for (int i = 0; i < n; i++) {
        d[i] = s[i] > thresh ? s[i] : 0;
    }
}

static void thresh_tozero_inv_row(const uchar *restrict s, uchar *restrict d,
                                  int n, uchar thresh, uchar maxval) {
    (void)maxval;
    for (int i = 0; i < n; i++) {
        d[i] = s[i] > thresh ? 0 : s[i];
    }
}

typedef void (*ThreshRowFn)(const uchar *restrict, uchar *restrict, int, uchar,
                            uchar);

static int is_u8_image_type(size_t t) {
    return t == (size_t)CV_8UC1 || t == (size_t)CV_8UC3 || t == (size_t)CV_8UC4;
}

void threshold(struct Mat *dst, const struct Mat *src, uchar thresh,
               uchar maxval, int type) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(is_u8_image_type(mat_type(src)));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);

    ThreshRowFn kernel = NULL;
    switch (type) {
    case THRESH_BINARY:
        kernel = thresh_binary_row;
        break;
    case THRESH_BINARY_INV:
        kernel = thresh_binary_inv_row;
        break;
    case THRESH_TRUNC:
        kernel = thresh_trunc_row;
        break;
    case THRESH_TOZERO:
        kernel = thresh_tozero_row;
        break;
    case THRESH_TOZERO_INV:
        kernel = thresh_tozero_inv_row;
        break;
    default:
        ASSERT(0 && "unknown threshold type");
    }

    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    int n = src->cols * (int)mat_channel_num(src);
    for (int y = 0; y < src->rows; y++) {
        kernel(mat_row(src, y), mat_row(dst, y), n, thresh, maxval);
    }
}

static void lut_row(const uchar *restrict s, uchar *restrict d, int n,
                    const uchar *lut) {
    for (int i = 0; i < n; i++) {
        d[i] = lut[s[i]];
    }
}

void mat_lut(struct Mat *dst, const struct Mat *src, const uchar lut[256]) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(lut != NULL);
    ASSERT(is_u8_image_type(mat_type(src)));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);

    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    int n = src->cols * (int)mat_channel_num(src);
    for (int y = 0; y < src->rows; y++) {
        lut_row(mat_row(src, y), mat_row(dst, y), n, lut);
    }
}

static void absdiff_row(const uchar *restrict a, const uchar *restrict b,
                        uchar *restrict d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        d[i] = (uchar)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
    }
}

void absdiff(struct Mat *dst, const struct Mat *a, const struct Mat *b) {
    ASSERT(dst != NULL && a != NULL && b != NULL);
    ASSERT(dst != a);
    ASSERT(dst != b);
    ASSERT(is_u8_image_type(mat_type(a)));
    ASSERT(mat_type(b) == mat_type(a) && mat_type(dst) == mat_type(a));
    ASSERT(b->rows == a->rows && b->cols == a->cols);
    ASSERT(dst->rows == a->rows && dst->cols == a->cols);

    if (a->rows == 0 || a->cols == 0) {
        return;
    }
    ASSERT(dst->data != a->data);
    ASSERT(dst->data != b->data);
    // size_t: cols * channels can exceed a 16-bit int.
    size_t n = (size_t)a->cols * mat_channel_num(a);
    for (int y = 0; y < a->rows; y++) {
        absdiff_row(mat_row(a, y), mat_row(b, y), mat_row(dst, y), n);
    }
}

// a / 256 rounded half to even and saturated to 0..255, for a >= 0. OpenCV
// rounds the float result with cvRound, which is half to even.
static inline uchar sat_q8_half_even(int32_t a) {
    int32_t q = a >> 8;
    int32_t r = a & 255;
    q += r > 128 || (r == 128 && (q & 1));
    return (uchar)(q > 255 ? 255 : q);
}

void convert_scale(struct Mat *dst, const struct Mat *src, int32_t alpha_q8,
                   int32_t beta) {
    ASSERT(dst != NULL && src != NULL);
    ASSERT(dst != src);
    ASSERT(is_u8_image_type(mat_type(src)));
    ASSERT(mat_type(dst) == mat_type(src));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(alpha_q8 >= -32768 && alpha_q8 <= 32767);
    ASSERT(beta >= -65536 && beta <= 65535);

    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT(dst->data != src->data);
    uchar lut[256];
    for (int i = 0; i < 256; i++) {
        int32_t v = (int32_t)i * alpha_q8 + beta * 256;
        lut[i]    = v < 0 ? 0 : sat_q8_half_even(v);
    }
    int n = src->cols * (int)mat_channel_num(src);
    for (int y = 0; y < src->rows; y++) {
        lut_row(mat_row(src, y), mat_row(dst, y), n, lut);
    }
}

static void scale_abs_row(const int16_t *restrict s, uchar *restrict d, int n,
                          int32_t alpha_q8, int32_t beta) {
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)s[i] * alpha_q8 + beta * 256;
        d[i]      = sat_q8_half_even(v < 0 ? -v : v);
    }
}

void convert_scale_abs(struct Mat *dst, const struct Mat *src, int32_t alpha_q8,
                       int32_t beta) {
    ASSERT(dst != NULL && src != NULL);
    size_t st = mat_type(src);
    ASSERT(st == (size_t)CV_16SC1 || st == (size_t)CV_16SC3 ||
           st == (size_t)CV_16SC4);
    ASSERT(mat_channel_num(dst) == mat_channel_num(src));
    ASSERT(mat_type(dst) ==
           (size_t)CV_MAKETYPE(CV_8U, (int)mat_channel_num(src)));
    ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    ASSERT(alpha_q8 >= -32768 && alpha_q8 <= 32767);
    ASSERT(beta >= -65536 && beta <= 65535);

    if (src->rows == 0 || src->cols == 0) {
        return;
    }
    ASSERT((const void *)dst->data != (const void *)src->data);
    int n = src->cols * (int)mat_channel_num(src);
    for (int y = 0; y < src->rows; y++) {
        scale_abs_row((const int16_t *)mat_row(src, y), mat_row(dst, y), n,
                      alpha_q8, beta);
    }
}
