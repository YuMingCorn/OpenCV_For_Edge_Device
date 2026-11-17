#include <stdlib.h>
#include <string.h>

#include "improc.h"
#include "kodak.h"

// SIFT runs on the half-size gray image: the full 768 x 512 image upsampled
// to 1536 x 1024 takes too long under qemu, and the half image still has
// several hundred keypoints.
#define KODAK_SIFT_MAX 4000 // more than any half-size Kodak image yields

// Rotation by 30 degrees and scale 0.75 about the image centre, Q16:
// forward 0.75 * (cos 30, sin 30), inverse (cos 30, sin 30) / 0.75, rounded.
#define FWD_COS_Q16 42567
#define FWD_SIN_Q16 24576
#define INV_COS_Q16 75674
#define INV_SIN_Q16 43691

static struct Mat *op_gray(const struct Mat *rgb) {
    struct Mat *out = alloc_mat(rgb->rows, rgb->cols, CV_8UC1);
    cvt_color(out, rgb, COLOR_RGB2GRAY);
    return out;
}

// Gray image halved with INTER_AREA; the input of every SIFT op.
static struct Mat *op_half(const struct Mat *rgb) {
    struct Mat *gray = op_gray(rgb);
    struct Mat *out  = alloc_mat(rgb->rows / 2, rgb->cols / 2, CV_8UC1);
    resize(out, gray, INTER_AREA);
    free_mat(&gray);
    return out;
}

// Maps (x, y), Q16 in and out, through the rotation and scale about the
// centre ((cols - 1) / 2, (rows - 1) / 2). inverse != 0 maps a destination
// point to its source. Integer only, so every platform gets the same bytes.
static void rs_map(int64_t *qx, int64_t *qy, int64_t x, int64_t y, int cols,
                   int rows, int inverse) {
    const int64_t c  = inverse ? INV_COS_Q16 : FWD_COS_Q16;
    const int64_t s  = inverse ? -INV_SIN_Q16 : FWD_SIN_Q16;
    const int64_t cx = (int64_t)(cols - 1) * 32768; // centre, Q16
    const int64_t cy = (int64_t)(rows - 1) * 32768;
    const int64_t dx = x - cx, dy = y - cy;
    *qx = (dx * c - dy * s) / 65536 + cx;
    *qy = (dx * s + dy * c) / 65536 + cy;
}

// The half image rotated by 30 degrees (clockwise on screen) and scaled by
// 0.75, bilinear; pixels whose source falls outside are 0.
static struct Mat *op_rotscale(const struct Mat *rgb) {
    struct Mat *half = op_half(rgb);
    struct Mat *out  = alloc_mat(half->rows, half->cols, CV_8UC1);
    for (int y = 0; y < out->rows; y++) {
        for (int x = 0; x < out->cols; x++) {
            int64_t sx, sy;
            rs_map(&sx, &sy, (int64_t)x * 65536, (int64_t)y * 65536, out->cols,
                   out->rows, 1);
            uchar v = 0;
            if (sx >= 0 && sy >= 0 && sx < (int64_t)(half->cols - 1) * 65536 &&
                sy < (int64_t)(half->rows - 1) * 65536) {
                const int     x0 = (int)(sx >> 16), y0 = (int)(sy >> 16);
                const int     fx  = (int)((sx >> 8) & 255);
                const int     fy  = (int)((sy >> 8) & 255);
                const uchar  *r0  = mat_row(half, y0) + x0;
                const uchar  *r1  = mat_row(half, y0 + 1) + x0;
                const int32_t top = r0[0] * (256 - fx) + r0[1] * fx;
                const int32_t bot = r1[0] * (256 - fx) + r1[1] * fx;
                v = (uchar)((top * (256 - fy) + bot * fy + 32768) >> 16);
            }
            mat_row(out, y)[x] = v;
        }
    }
    free_mat(&half);
    return out;
}

static struct SiftKeyPoint kps_a[KODAK_SIFT_MAX], kps_b[KODAK_SIFT_MAX];
static uint8_t             desc_a[KODAK_SIFT_MAX][SIFT_DESC_BYTES];
static uint8_t             desc_b[KODAK_SIFT_MAX][SIFT_DESC_BYTES];

// N x 7 CV_32F: x, y, size, angle, response, octave, layer of the SIFT
// keypoints of the half image (default parameters).
static struct Mat *op_sift_kps(const struct Mat *rgb) {
    struct Mat   *half = op_half(rgb);
    const int32_t n =
        sift_detect_compute(kps_a, NULL, KODAK_SIFT_MAX, half, NULL);
    struct Mat *out = alloc_mat(n > 0 ? n : 1, 7, CV_32FC1);
    memset(out->data, 0, (size_t)out->rows * out->step[0]);
    for (int32_t i = 0; i < n; i++) {
        float *r = (float *)mat_row(out, i);
        r[0]     = kps_a[i].x;
        r[1]     = kps_a[i].y;
        r[2]     = kps_a[i].size;
        r[3]     = kps_a[i].angle;
        r[4]     = kps_a[i].response;
        r[5]     = (float)kps_a[i].octave;
        r[6]     = (float)kps_a[i].layer;
    }
    free_mat(&half);
    return out;
}

// N x 128 CV_8UC1: the descriptors of the sift_kps keypoints, row i for
// keypoint i.
static struct Mat *op_sift_desc(const struct Mat *rgb) {
    struct Mat   *half = op_half(rgb);
    const int32_t n =
        sift_detect_compute(kps_a, desc_a, KODAK_SIFT_MAX, half, NULL);
    struct Mat *out = alloc_mat(n > 0 ? n : 1, SIFT_DESC_BYTES, CV_8UC1);
    memset(out->data, 0, (size_t)out->rows * out->step[0]);
    for (int32_t i = 0; i < n; i++) {
        memcpy(mat_row(out, i), desc_a[i], SIFT_DESC_BYTES);
    }
    free_mat(&half);
    return out;
}

// 1 x 3 CV_32S: [ratio-test (0.8) matches between half and rotscale, matches
// that land within 3 pixels of the true position, sum of the half image's
// descriptor bytes]. tools/opencv_ref/sift.py compares the correct share
// with cv2.SIFT on the same two images.
static struct Mat *op_sift_rotscale(const struct Mat *rgb) {
    static struct SiftMatch mt[KODAK_SIFT_MAX];
    struct Mat             *half = op_half(rgb);
    struct Mat             *rot  = op_rotscale(rgb);
    const int32_t           na =
        sift_detect_compute(kps_a, desc_a, KODAK_SIFT_MAX, half, NULL);
    const int32_t nb =
        sift_detect_compute(kps_b, desc_b, KODAK_SIFT_MAX, rot, NULL);
    const int32_t nm   = sift_match(mt, desc_a, na, desc_b, nb, 0.8f);
    int32_t       good = 0;
    for (int32_t i = 0; i < nm; i++) {
        const struct SiftKeyPoint *p = &kps_a[mt[i].query];
        const struct SiftKeyPoint *q = &kps_b[mt[i].train];
        int64_t                    ex, ey;
        rs_map(&ex, &ey, (int64_t)(p->x * 65536.0f), (int64_t)(p->y * 65536.0f),
               half->cols, half->rows, 0);
        ex -= (int64_t)(q->x * 65536.0f);
        ey -= (int64_t)(q->y * 65536.0f);
        good += ex * ex + ey * ey < (int64_t)9 * 65536 * 65536;
    }
    int32_t sum = 0;
    for (int32_t i = 0; i < na; i++) {
        for (int k = 0; k < SIFT_DESC_BYTES; k++) {
            sum += desc_a[i][k];
        }
    }
    struct Mat *out           = alloc_mat(1, 3, CV_MAKETYPE(CV_32S, 1));
    ((int32_t *)out->data)[0] = nm;
    ((int32_t *)out->data)[1] = good;
    ((int32_t *)out->data)[2] = sum;
    free_mat(&half);
    free_mat(&rot);
    return out;
}

const struct KodakOp kodak_ops[] = {
    {"gray", op_gray},           {"half", op_half},
    {"rotscale", op_rotscale},   {"sift_kps", op_sift_kps},
    {"sift_desc", op_sift_desc}, {"sift_rotscale", op_sift_rotscale},
};

const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
