// Times every image op on one photo and prints ns per source pixel together
// with a 64-bit FNV-1a hash of the output, so a faster version can be checked
// byte for byte against the old one.
//
//   bench_ops <image.png> [--reps N] [--only NAME]
//
// Each op runs in batches; the reported time is the median of 9 batches, in
// nanoseconds per source pixel (rows * cols of the input, not per channel).
// On a hybrid CPU pin the process to one core (Windows: start /affinity 1).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "improc.h"

static double now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t fnv(uint64_t h, const uchar *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t hash_mat(const struct Mat *m) {
    uint64_t h   = 14695981039346656037ULL;
    size_t   row = (size_t)m->cols * mat_pixel_size(m);
    for (int y = 0; y < m->rows; y++) {
        h = fnv(h, mat_row(m, y), row);
    }
    return h;
}

// Inputs shared by all ops.
static struct Mat *rgb, *gray, *hsv, *yuv, *s16, *c2, *gray2;

struct Op {
    const char *name;
    // Runs the op once into *out (allocated on the first call).
    void (*run)(struct Mat **out);
};

static struct Mat *need(struct Mat **out, int rows, int cols, int type) {
    if (*out == NULL) {
        *out = alloc_mat(rows, cols, type);
    }
    return *out;
}

#define R rgb->rows
#define C rgb->cols

static void op_gray(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC1), rgb, COLOR_RGB2GRAY);
}
static void op_bgr(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), rgb, COLOR_RGB2BGR);
}
static void op_rgb2hsv(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), rgb, COLOR_RGB2HSV);
}
static void op_hsv2rgb(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), hsv, COLOR_HSV2RGB);
}
static void op_rgb2yuv(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), rgb, COLOR_RGB2YUV);
}
static void op_yuv2rgb(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), yuv, COLOR_YUV2RGB);
}
static void op_565gray(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC1), c2, COLOR_BGR5652GRAY);
}
static void op_565rgb(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), c2, COLOR_BGR5652RGB);
}
static void op_yuyv_rgb(struct Mat **o) {
    cvt_color(need(o, R, C, CV_8UC3), c2, COLOR_YUV2RGB_YUYV);
}
static void op_thresh(struct Mat **o) {
    threshold(need(o, R, C, CV_8UC3), rgb, 128, 255, THRESH_BINARY);
}
static void op_scale(struct Mat **o) {
    convert_scale(need(o, R, C, CV_8UC3), rgb, 384, 10);
}
static void op_scale_abs(struct Mat **o) {
    convert_scale_abs(need(o, R, C, CV_8UC1), s16, 256, 0);
}
static void op_absdiff(struct Mat **o) {
    absdiff(need(o, R, C, CV_8UC1), gray, gray2);
}
static void op_box5(struct Mat **o) {
    box_filter(need(o, R, C, CV_8UC3), rgb, 5);
}
static void op_gauss3(struct Mat **o) {
    gaussian_blur(need(o, R, C, CV_8UC3), rgb, 3);
}
static void op_gauss5(struct Mat **o) {
    gaussian_blur(need(o, R, C, CV_8UC3), rgb, 5);
}
static void op_gauss7(struct Mat **o) {
    gaussian_blur(need(o, R, C, CV_8UC3), rgb, 7);
}
static void op_sobel_x(struct Mat **o) {
    sobel(need(o, R, C, CV_16SC1), gray, 1, 0);
}
static void op_median3(struct Mat **o) {
    median_blur(need(o, R, C, CV_8UC3), rgb, 3);
}
static void op_median5(struct Mat **o) {
    median_blur(need(o, R, C, CV_8UC3), rgb, 5);
}
static void op_median7_gray(struct Mat **o) {
    median_blur(need(o, R, C, CV_8UC1), gray, 7);
}
static void op_erode3(struct Mat **o) { erode(need(o, R, C, CV_8UC3), rgb, 3); }
static void op_dilate3(struct Mat **o) {
    dilate(need(o, R, C, CV_8UC3), rgb, 3);
}
static void op_open5(struct Mat **o) {
    static struct Mat *tmp;
    morph_open(need(o, R, C, CV_8UC3), need(&tmp, R, C, CV_8UC3), rgb, 5);
}
static void op_equalize(struct Mat **o) {
    equalize_hist(need(o, R, C, CV_8UC1), gray);
}
static void op_flip_h(struct Mat **o) {
    flip(need(o, R, C, CV_8UC3), rgb, FLIP_HORIZONTAL);
}
static void op_linear_nonint(struct Mat **o) {
    resize(need(o, R * 5 / 12 + 1, C * 3 / 7, CV_8UC3), rgb, INTER_LINEAR);
}
static void op_linear_up(struct Mat **o) {
    resize(need(o, R * 3 / 2, C * 3 / 2, CV_8UC3), rgb, INTER_LINEAR);
}
static void op_area_half(struct Mat **o) {
    resize(need(o, R / 2, C / 2, CV_8UC3), rgb, INTER_AREA);
}
static void op_area_third(struct Mat **o) {
    resize(need(o, R / 3, C / 3, CV_8UC3), rgb, INTER_AREA);
}
static void op_area_up(struct Mat **o) {
    resize(need(o, R * 3 / 2, C * 3 / 2, CV_8UC3), rgb, INTER_AREA);
}
static void op_nearest_up(struct Mat **o) {
    resize(need(o, R * 2, C * 2, CV_8UC3), rgb, INTER_NEAREST);
}

// fast_detect returns key points, not an image: hash them as bytes.
static void op_fast(struct Mat **o) {
    enum { CAP = 1 << 16 };
    struct Mat *m = need(o, 1, CAP * (int)sizeof(struct KeyPoint), CV_8UC1);
    int32_t     n = fast_detect((struct KeyPoint *)m->data, CAP, gray, 20, 1);
    memset(m->data + (size_t)n * sizeof(struct KeyPoint), 0,
           (size_t)(CAP - n) * sizeof(struct KeyPoint));
}

static const struct Op ops[] = {
    {"gray", op_gray},
    {"bgr", op_bgr},
    {"rgb2hsv", op_rgb2hsv},
    {"hsv2rgb", op_hsv2rgb},
    {"rgb2yuv", op_rgb2yuv},
    {"yuv2rgb", op_yuv2rgb},
    {"565gray", op_565gray},
    {"565rgb", op_565rgb},
    {"yuyv_rgb", op_yuyv_rgb},
    {"thresh", op_thresh},
    {"scale", op_scale},
    {"scale_abs", op_scale_abs},
    {"absdiff", op_absdiff},
    {"box5", op_box5},
    {"gauss3", op_gauss3},
    {"gauss5", op_gauss5},
    {"gauss7", op_gauss7},
    {"sobel_x", op_sobel_x},
    {"median3", op_median3},
    {"median5", op_median5},
    {"median7_gray", op_median7_gray},
    {"erode3", op_erode3},
    {"dilate3", op_dilate3},
    {"open5", op_open5},
    {"equalize", op_equalize},
    {"flip_h", op_flip_h},
    {"linear_nonint", op_linear_nonint},
    {"linear_up", op_linear_up},
    {"area_half", op_area_half},
    {"area_third", op_area_third},
    {"area_up", op_area_up},
    {"nearest_up", op_nearest_up},
    {"fast", op_fast},
};

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *only = NULL;
    int         reps = 0;
    if (argc < 2) {
        fprintf(stderr,
                "usage: bench_ops <image.png> [--reps N] [--only NAME]\n");
        return 2;
    }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 < argc && strcmp(argv[i], "--reps") == 0) {
            reps = atoi(argv[i + 1]);
        } else if (i + 1 < argc && strcmp(argv[i], "--only") == 0) {
            only = argv[i + 1];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (only != NULL) {
        int found = 0;
        for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++) {
            found |= strcmp(only, ops[k].name) == 0;
        }
        if (!found) {
            fprintf(stderr, "no op named %s\n", only);
            return 2;
        }
    }
    rgb = alloc_mat_from_file(argv[1]);
    if (rgb == NULL || mat_type(rgb) != CV_8UC3) {
        fprintf(stderr, "cannot read an RGB image from %s\n", argv[1]);
        return 2;
    }
    gray = alloc_mat(R, C, CV_8UC1);
    cvt_color(gray, rgb, COLOR_RGB2GRAY);
    hsv = alloc_mat(R, C, CV_8UC3);
    cvt_color(hsv, rgb, COLOR_RGB2HSV);
    yuv = alloc_mat(R, C, CV_8UC3);
    cvt_color(yuv, rgb, COLOR_RGB2YUV);
    s16 = alloc_mat(R, C, CV_16SC1);
    sobel(s16, gray, 1, 0);
    // Two bytes per pixel from the photo, for the 565 and YUYV decoders.
    c2 = alloc_mat(R, C, CV_8UC2);
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            const uchar *p = mat_at(rgb, y, x);
            uchar       *q = mat_at(c2, y, x);
            q[0]           = p[0];
            q[1]           = p[1];
        }
    }
    // gray shifted by 6 columns, for absdiff.
    gray2 = alloc_mat(R, C, CV_8UC1);
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            *(uchar *)mat_at(gray2, y, x) =
                *(uchar *)mat_at(gray, y, (x + 6) % C);
        }
    }

    double px = (double)R * (double)C;
    printf("%-14s %10s  %s\n", "op", "ns/px", "hash");
    for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++) {
        if (only != NULL && strcmp(only, ops[k].name) != 0) {
            continue;
        }
        struct Mat *out = NULL;
        ops[k].run(&out); // warm up, allocate
        int n = reps;
        if (n <= 0) { // aim for about 20 ms per batch
            double t0 = now_ns();
            ops[k].run(&out);
            double one = now_ns() - t0;
            n          = one > 2e7 ? 1 : (int)(2e7 / (one + 1.0)) + 1;
        }
        double t[9];
        for (int b = 0; b < 9; b++) {
            double t0 = now_ns();
            for (int i = 0; i < n; i++) {
                ops[k].run(&out);
            }
            t[b] = (now_ns() - t0) / n / px;
        }
        qsort(t, 9, sizeof t[0], cmp_double);
        printf("%-14s %10.3f  %016llx\n", ops[k].name, t[4],
               (unsigned long long)hash_mat(out));
        free_mat(&out);
    }
    return 0;
}
