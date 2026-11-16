#include <stdlib.h>
#include <string.h>

#include "improc.h"
#include "kodak.h"

// FAST threshold used by both ops; the cv2 default is 10, 20 keeps the maps
// sparse enough to read.
#define KODAK_FAST_THRESHOLD 20

// Project gray image; the input of the other ops in this module.
static struct Mat *op_gray(const struct Mat *rgb) {
    struct Mat *out = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    cvt_color(out, rgb, COLOR_RGB2GRAY);
    return out;
}

// CV_8UC1 map of the FAST corners of the gray image: value at each corner,
// 0 elsewhere. With nonmax the value is the response (always >= 1 for a kept
// corner); without it, 255.
static struct Mat *fast_map(const struct Mat *rgb, int nonmax) {
    struct Mat   *gray = op_gray(rgb);
    struct Mat   *out  = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    const int32_t n = fast_detect(NULL, 0, gray, KODAK_FAST_THRESHOLD, nonmax);
    struct KeyPoint *kps = malloc(sizeof *kps * (size_t)(n > 0 ? n : 1));
    if (kps == NULL) {
        abort();
    }
    fast_detect(kps, n, gray, KODAK_FAST_THRESHOLD, nonmax);
    memset(out->data, 0, (size_t)out->rows * out->step[0]);
    for (int32_t i = 0; i < n; i++) {
        mat_row(out, kps[i].y)[kps[i].x] =
            (uchar)(nonmax ? kps[i].response : 255);
    }
    free(kps);
    free_mat(&gray);
    return out;
}

static struct Mat *op_fast(const struct Mat *rgb) { return fast_map(rgb, 1); }

static struct Mat *op_fast_raw(const struct Mat *rgb) {
    return fast_map(rgb, 0);
}

const struct KodakOp kodak_ops[] = {
    {"gray", op_gray},
    {"fast", op_fast},
    {"fast_raw", op_fast_raw},
};

const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
