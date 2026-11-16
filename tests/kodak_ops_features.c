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

// ---- ORB --------------------------------------------------------------------

#define KODAK_ORB_KPS 500 // the cv2.ORB_create default
// Rotation of the rot30 op, Q16: cos 30 and sin 30 degrees, rounded.
#define ROT_COS_Q16 56756
#define ROT_SIN_Q16 32768

// Maps (x, y) through the 30-degree rotation about the image centre
// ((cols - 1) / 2, (rows - 1) / 2). inverse != 0 maps a destination pixel to
// its source. Results are Q16, computed on doubled coordinates so the centre
// stays an integer. Integer only, so every platform gets the same bytes.
static void rot30_map(int64_t *qx, int64_t *qy, int64_t x, int64_t y, int cols,
                      int rows, int inverse) {
    const int64_t dx = 2 * x - (cols - 1);
    const int64_t dy = 2 * y - (rows - 1);
    const int64_t s  = inverse ? -ROT_SIN_Q16 : ROT_SIN_Q16;
    const int64_t rx = dx * ROT_COS_Q16 - dy * s;
    const int64_t ry = dx * s + dy * ROT_COS_Q16;
    *qx              = (rx + (int64_t)(cols - 1) * 65536) / 2;
    *qy              = (ry + (int64_t)(rows - 1) * 65536) / 2;
}

// The gray image rotated by 30 degrees (clockwise on screen, since y points
// down: a source offset (1, 0) lands at (cos 30, sin 30)) with
// bilinear sampling; pixels whose source falls outside are 0.
static struct Mat *op_rot30(const struct Mat *rgb) {
    struct Mat *gray = op_gray(rgb);
    struct Mat *out  = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    for (int y = 0; y < out->rows; y++) {
        for (int x = 0; x < out->cols; x++) {
            int64_t sx, sy;
            rot30_map(&sx, &sy, x, y, out->cols, out->rows, 1);
            uchar v = 0;
            if (sx >= 0 && sy >= 0 && sx < (int64_t)(gray->cols - 1) * 65536 &&
                sy < (int64_t)(gray->rows - 1) * 65536) {
                const int     x0 = (int)(sx >> 16), y0 = (int)(sy >> 16);
                const int     fx  = (int)((sx >> 8) & 255);
                const int     fy  = (int)((sy >> 8) & 255);
                const uchar  *r0  = mat_row(gray, y0) + x0;
                const uchar  *r1  = mat_row(gray, y0 + 1) + x0;
                const int32_t top = r0[0] * (256 - fx) + r0[1] * fx;
                const int32_t bot = r1[0] * (256 - fx) + r1[1] * fx;
                v = (uchar)((top * (256 - fy) + bot * fy + 32768) >> 16);
            }
            mat_row(out, y)[x] = v;
        }
    }
    free_mat(&gray);
    return out;
}

// ORB keypoints and descriptors of a gray image (blurred with ksize 7 for
// sampling). Returns the count.
static int32_t orb_run(struct KeyPoint  *kps, uint8_t (*desc)[ORB_DESC_BYTES],
                       const struct Mat *gray) {
    struct Mat   *smooth = alloc_mat(gray->rows, gray->cols, CV_8UC1);
    const int32_t n =
        orb_detect(kps, KODAK_ORB_KPS, gray, KODAK_FAST_THRESHOLD);
    gaussian_blur(smooth, gray, 7);
    orb_compute(desc, kps, n, smooth);
    free_mat(&smooth);
    return n;
}

// CV_8UC1 map of the ORB keypoints of the gray image: angle * 8 + 4 at each
// keypoint, 0 elsewhere. Pins orb_detect through the golden hash.
static struct Mat *op_orb_map(const struct Mat *rgb) {
    static struct KeyPoint kps[KODAK_ORB_KPS];
    static uint8_t         desc[KODAK_ORB_KPS][ORB_DESC_BYTES];
    struct Mat            *gray = op_gray(rgb);
    struct Mat            *out  = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    const int32_t          n    = orb_run(kps, desc, gray);
    memset(out->data, 0, (size_t)out->rows * out->step[0]);
    for (int32_t i = 0; i < n; i++) {
        mat_row(out, kps[i].y)[kps[i].x] = (uchar)(kps[i].angle * 8 + 4);
    }
    free_mat(&gray);
    return out;
}

// 1 x 3 CV_32S: [cross-checked matches between gray and rot30, matches that
// land within 3 pixels of the true rotated position, sum of the descriptor
// bytes]. tools/opencv_ref/features.py compares the correct share with
// cv2.ORB on the same two images; the sum pins orb_compute in the hash.
static struct Mat *op_orb_rot30(const struct Mat *rgb) {
    static struct KeyPoint ka[KODAK_ORB_KPS], kb[KODAK_ORB_KPS];
    static uint8_t         da[KODAK_ORB_KPS][ORB_DESC_BYTES];
    static uint8_t         db[KODAK_ORB_KPS][ORB_DESC_BYTES];
    static struct DMatch   mt[KODAK_ORB_KPS];
    struct Mat            *gray = op_gray(rgb);
    struct Mat            *rot  = op_rot30(rgb);
    const int32_t          na   = orb_run(ka, da, gray);
    const int32_t          nb   = orb_run(kb, db, rot);
    const int32_t          nm   = match_descriptors(mt, da, na, db, nb, 1);
    int32_t                good = 0;
    for (int32_t i = 0; i < nm; i++) {
        const struct KeyPoint *p = &ka[mt[i].query];
        const struct KeyPoint *q = &kb[mt[i].train];
        int64_t                ex, ey;
        rot30_map(&ex, &ey, p->x, p->y, gray->cols, gray->rows, 0);
        ex -= (int64_t)q->x * 65536;
        ey -= (int64_t)q->y * 65536;
        good += ex * ex + ey * ey < (int64_t)9 * 65536 * 65536;
    }
    int32_t sum = 0;
    for (int32_t i = 0; i < na; i++) {
        for (int k = 0; k < ORB_DESC_BYTES; k++) {
            sum += da[i][k];
        }
    }
    struct Mat *out           = alloc_mat(1, 3, CV_MAKETYPE(CV_32S, 1));
    ((int32_t *)out->data)[0] = nm;
    ((int32_t *)out->data)[1] = good;
    ((int32_t *)out->data)[2] = sum;
    free_mat(&gray);
    free_mat(&rot);
    return out;
}

const struct KodakOp kodak_ops[] = {
    {"gray", op_gray},   {"fast", op_fast},       {"fast_raw", op_fast_raw},
    {"rot30", op_rot30}, {"orb_map", op_orb_map}, {"orb_rot30", op_orb_rot30},
};

const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
