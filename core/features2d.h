// Named features2d.h, not features.h: glibc's <stdint.h> includes its own
// <features.h>, and -Icore would make it find this file instead.
#ifndef FEATURES2D_H_
#define FEATURES2D_H_

#include <stdint.h>

#include "mat.h"

// One detected point. x is the column, y the row, both in pixels of the image
// that was searched.
struct KeyPoint {
    int32_t x;
    int32_t y;
    int32_t response; // FAST score; fast_detect without nonmax: 0
    int32_t angle;    // orb_detect: step 0..ORB_ANGLE_STEPS-1 of
                      // 360 / ORB_ANGLE_STEPS degrees, y down; fast_detect: -1
};

// FAST-9 on a CV_8UC1 image: the same points, order and response as
// cv2.FastFeatureDetector (TYPE_9_16). Pixels within 3 of an edge are not
// tested. With nonmax != 0 the score (0..254) is the largest threshold that
// still passes, and a corner is kept only when its score is strictly greater
// than all 8 neighbours.
//
// Writes the first min(total, capacity) points in raster order and returns
// total, which may exceed capacity; capacity 0 (kps may be NULL) only counts.
// threshold is 0..255.
// Scratch: 3 * cols bytes with nonmax, none without.
int32_t fast_detect(struct KeyPoint *kps, int32_t capacity,
                    const struct Mat *src, int threshold, int nonmax);

// ORB, single scale, integer only.
//
// Not bit-compatible with cv2.ORB: own seeded test pairs, and the angle is
// quantized to ORB_ANGLE_STEPS steps. Matching quality is on par with
// cv2.ORB_create(nlevels=1); see docs/kodak-results.md.

#define ORB_DESC_BYTES  32 // 256-bit descriptor
#define ORB_ANGLE_STEPS 32 // 11.25 degrees per step
#define ORB_BORDER      19 // keypoints keep this distance from every edge

// Detects up to max_kps oriented keypoints in a CV_8UC1 image: FAST with
// non-max suppression, corners at least ORB_BORDER inside, the 2 * max_kps
// best FAST scores re-ranked by Harris (k = 0.04, 7x7 window) as cv2.ORB does,
// and the angle of the intensity centroid of the radius-15 disc. Writes at
// most max_kps points in rank order (ties in raster order) and returns their
// count, 0 when rows or cols <= 2 * ORB_BORDER.
// Scratch: 3 * cols + n * (sizeof(struct KeyPoint) + 16) bytes for n FAST
// corners (+ 12 instead of + 16 on AVR).
int32_t orb_detect(struct KeyPoint *kps, int32_t max_kps, const struct Mat *src,
                   int threshold);

// One 256-bit rotated BRIEF descriptor per keypoint, sampled from smooth
// (CV_8UC1, normally gaussian_blur 5 or 7). Each keypoint must lie ORB_BORDER
// inside and have an angle in range (ASSERT). Bit i (byte i / 8, bit i % 8)
// is 1 when the first point of pair i, rotated, is darker than the second. No
// scratch.
void orb_compute(uint8_t (*desc)[ORB_DESC_BYTES], const struct KeyPoint *kps,
                 int32_t n, const struct Mat *smooth);

// Number of differing bits between two descriptors (0..256).
int hamming_distance(const uint8_t a[ORB_DESC_BYTES],
                     const uint8_t b[ORB_DESC_BYTES]);

struct DMatch {
    int32_t query;    // index into the query descriptors
    int32_t train;    // index into the train descriptors
    int32_t distance; // Hamming distance
};

// Brute force: for each query, the nearest train descriptor. cross_check != 0
// also requires the query to be the train's best match. Ties go to the lowest
// index. matches needs room for nq; returns the count (<= nq) in query order.
// Scratch: nt * 4 bytes with cross_check, none without.
int32_t match_descriptors(struct DMatch *matches,
                          const uint8_t (*query)[ORB_DESC_BYTES], int32_t nq,
                          const uint8_t (*train)[ORB_DESC_BYTES], int32_t nt,
                          int cross_check);

#endif // FEATURES2D_H_
