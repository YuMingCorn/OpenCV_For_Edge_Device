#include "improc.h"
#include "kodak.h"

#include <string.h>

// Edge-device motion pipeline on one Kodak image:
// crop -> absdiff -> threshold -> nonzero_stats -> draw the decision.
//
// Frame 0 is the image without its last SHIFT rows and columns. Frame 1 is a
// copy of frame 0 in which one patch (the "moving object") is replaced by the
// content SHIFT pixels down and right. Three patches cover the three cases of
// nonzero_stats: an interior box (motion_*), a box in the top-left corner whose
// centroid cross is clipped by the border (corner_*), and no motion at all
// (still_stats, count 0). The OpenCV references are in
// tools/opencv_ref/edge.py.

#define SHIFT       6
#define MOTION_T    40 // gray difference above this counts as motion
#define CROSS_HALF  6  // centroid cross arm length in pixels
#define CORNER_SIZE 12 // side of the top-left patch

struct Patch {
    int x, y, w, h;
};

static struct Mat *op_gray(const struct Mat *rgb) {
    struct Mat *out = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    cvt_color(out, rgb, COLOR_RGB2GRAY);
    return out;
}

// The middle half of the image in both directions.
static struct Mat *op_crop_center(const struct Mat *rgb) {
    struct Mat *out = alloc_mat(rgb->rows / 2, rgb->cols / 2, CV_8UC3);
    crop(out, rgb, rgb->cols / 4, rgb->rows / 4);
    return out;
}

// The middle half of frame 0.
static struct Patch center_patch(const struct Mat *src) {
    int          w = src->cols - SHIFT, h = src->rows - SHIFT;
    struct Patch p = {w / 4, h / 4, w / 2, h / 2};
    return p;
}

static struct Patch corner_patch(void) {
    struct Patch p = {0, 0, CORNER_SIZE, CORNER_SIZE};
    return p;
}

static struct Patch no_patch(void) {
    struct Patch p = {0, 0, 0, 0};
    return p;
}

// Frame 0 and frame 1 as described at the top of the file.
static void frames(const struct Mat *src, struct Patch p, struct Mat **f0,
                   struct Mat **f1) {
    *f0 = alloc_mat(src->rows - SHIFT, src->cols - SHIFT, mat_type(src));
    *f1 = alloc_mat(src->rows - SHIFT, src->cols - SHIFT, mat_type(src));
    crop(*f0, src, 0, 0);
    crop(*f1, src, 0, 0);
    if (p.w == 0 || p.h == 0) {
        return;
    }
    struct Mat *moved = alloc_mat(p.h, p.w, mat_type(src));
    crop(moved, src, p.x + SHIFT, p.y + SHIFT);
    size_t px = mat_pixel_size(src);
    for (int y = 0; y < p.h; y++) {
        memcpy(mat_row(*f1, p.y + y) + (size_t)p.x * px, mat_row(moved, y),
               (size_t)p.w * px);
    }
    free_mat(&moved);
}

// Whole-frame RGB difference: frame 1 is entirely the shifted image here.
static struct Mat *op_absdiff_shift(const struct Mat *rgb) {
    int          w = rgb->cols - SHIFT, h = rgb->rows - SHIFT;
    struct Patch all = {0, 0, w, h};
    struct Mat  *f0, *f1;
    frames(rgb, all, &f0, &f1);
    struct Mat *out = alloc_mat(f0->rows, f0->cols, CV_8UC3);
    absdiff(out, f0, f1);
    free_mat(&f0);
    free_mat(&f1);
    return out;
}

static struct Mat *motion_mask(const struct Mat *rgb, struct Patch p) {
    struct Mat *gray = op_gray(rgb);
    struct Mat *f0, *f1;
    frames(gray, p, &f0, &f1);
    struct Mat *diff = alloc_mat(f0->rows, f0->cols, CV_8UC1);
    struct Mat *mask = alloc_mat(f0->rows, f0->cols, CV_8UC1);
    absdiff(diff, f0, f1);
    threshold(mask, diff, MOTION_T, 255, THRESH_BINARY);
    free_mat(&diff);
    free_mat(&f0);
    free_mat(&f1);
    free_mat(&gray);
    return mask;
}

static struct Mat *op_motion_mask(const struct Mat *rgb) {
    return motion_mask(rgb, center_patch(rgb));
}

// Rounded centroid, half up; only valid when count > 0.
static int centroid(uint64_t sum, uint32_t count) {
    return (int)((sum + count / 2) / count);
}

// 1 x 7 CV_32S: count, x_min, y_min, x_max, y_max, centroid x, centroid y.
// The centroid is -1, -1 when no pixel moved.
static struct Mat *stats_of(const struct Mat *rgb, struct Patch p) {
    struct Mat         *mask = motion_mask(rgb, p);
    struct NonzeroStats s;
    nonzero_stats(&s, mask);
    struct Mat *out = alloc_mat(1, 7, CV_MAKETYPE(CV_32S, 1));
    int32_t    *v   = (int32_t *)out->data;
    v[0]            = (int32_t)s.count;
    v[1]            = s.x_min;
    v[2]            = s.y_min;
    v[3]            = s.x_max;
    v[4]            = s.y_max;
    v[5]            = s.count ? centroid(s.sum_x, s.count) : -1;
    v[6]            = s.count ? centroid(s.sum_y, s.count) : -1;
    free_mat(&mask);
    return out;
}

static struct Mat *op_motion_stats(const struct Mat *rgb) {
    return stats_of(rgb, center_patch(rgb));
}

static struct Mat *op_corner_stats(const struct Mat *rgb) {
    return stats_of(rgb, corner_patch());
}

static struct Mat *op_still_stats(const struct Mat *rgb) {
    return stats_of(rgb, no_patch());
}

static void put(struct Mat *m, int x, int y, uchar r, uchar g, uchar b) {
    if (x < 0 || y < 0 || x >= m->cols || y >= m->rows) {
        return;
    }
    uchar *q = mat_row(m, y) + 3 * (size_t)x;
    q[0]     = r;
    q[1]     = g;
    q[2]     = b;
}

// Frame 1 with the motion bounding box (red, 1 px) and the centroid (green
// cross, clipped at the border) drawn on it: the decision an edge device acts
// on.
static struct Mat *box_of(const struct Mat *rgb, struct Patch p) {
    struct Mat         *mask = motion_mask(rgb, p);
    struct NonzeroStats s;
    nonzero_stats(&s, mask);
    struct Mat *f0, *out;
    frames(rgb, p, &f0, &out);
    free_mat(&f0);
    if (s.count > 0) {
        for (int x = s.x_min; x <= s.x_max; x++) {
            put(out, x, s.y_min, 255, 0, 0);
            put(out, x, s.y_max, 255, 0, 0);
        }
        for (int y = s.y_min; y <= s.y_max; y++) {
            put(out, s.x_min, y, 255, 0, 0);
            put(out, s.x_max, y, 255, 0, 0);
        }
        int cx = centroid(s.sum_x, s.count);
        int cy = centroid(s.sum_y, s.count);
        for (int d = -CROSS_HALF; d <= CROSS_HALF; d++) {
            put(out, cx + d, cy, 0, 255, 0);
            put(out, cx, cy + d, 0, 255, 0);
        }
    }
    free_mat(&mask);
    return out;
}

static struct Mat *op_motion_box(const struct Mat *rgb) {
    return box_of(rgb, center_patch(rgb));
}

static struct Mat *op_corner_box(const struct Mat *rgb) {
    return box_of(rgb, corner_patch());
}

const struct KodakOp kodak_ops[] = {
    {"edge_gray", op_gray},
    {"crop_center", op_crop_center},
    {"absdiff_shift", op_absdiff_shift},
    {"motion_mask", op_motion_mask},
    {"motion_stats", op_motion_stats},
    {"motion_box", op_motion_box},
    {"corner_stats", op_corner_stats},
    {"corner_box", op_corner_box},
    {"still_stats", op_still_stats},
};
const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
