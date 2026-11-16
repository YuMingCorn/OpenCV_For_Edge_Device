#include "improc.h"
#include "kodak.h"

static struct Mat *median_rgb(const struct Mat *rgb, int ksize) {
    struct Mat *out = alloc_mat(rgb->rows, rgb->cols, CV_8UC3);
    median_blur(out, rgb, ksize);
    return out;
}

static struct Mat *op_median3(const struct Mat *rgb) {
    return median_rgb(rgb, 3);
}

static struct Mat *op_median5(const struct Mat *rgb) {
    return median_rgb(rgb, 5);
}

// Project gray image, dumped so OpenCV filters exactly the same input.
static struct Mat *op_median_gray_src(const struct Mat *rgb) {
    struct Mat *gray = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    cvt_color(gray, rgb, COLOR_RGB2GRAY);
    return gray;
}

static struct Mat *median_gray(const struct Mat *rgb, int ksize) {
    struct Mat *gray = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    struct Mat *out  = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    cvt_color(gray, rgb, COLOR_RGB2GRAY);
    median_blur(out, gray, ksize);
    free_mat(&gray);
    return out;
}

static struct Mat *op_gray_median7(const struct Mat *rgb) {
    return median_gray(rgb, 7);
}

static struct Mat *op_gray_median15(const struct Mat *rgb) {
    return median_gray(rgb, 15);
}

const struct KodakOp kodak_ops[] = {
    {"median3", op_median3},
    {"median5", op_median5},
    {"median_gray_src", op_median_gray_src},
    {"gray_median7", op_gray_median7},
    {"gray_median15", op_gray_median15},
};

const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
