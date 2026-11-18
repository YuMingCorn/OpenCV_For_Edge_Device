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
    int32_t response; // fast_detect: corner score with non-max suppression,
                      // 0 without it (as cv2.FAST)
    int32_t angle;    // orb_detect: orientation step s in
                      // 0..ORB_ANGLE_STEPS-1, the direction
                      // (cos, sin)(s * 360 / ORB_ANGLE_STEPS degrees) with x to
                      // the right and y down; fast_detect: -1
};

// FAST-9 corner detector on a CV_8UC1 image, the same result as
// cv2.FastFeatureDetector (TYPE_9_16): same points, same order, same response.
//
// Pixel p is a corner when 9 contiguous pixels of the radius-3 Bresenham
// circle (16 pixels) are all brighter than p + threshold, or all darker than
// p - threshold (strict compare). Only pixels with 3 <= x <= cols - 4 and
// 3 <= y <= rows - 4 are tested, so an image smaller than 7x7 has no corner.
//
// With nonmax != 0 the score of a corner is the largest threshold for which it
// is still a corner, minus 1 (0..254), and a corner is kept only when its score
// is strictly greater than the scores of all 8 neighbours (a pixel that is not
// a corner counts as 0). Equal neighbours therefore suppress each other.
//
// Writes the first min(total, capacity) points to kps in raster order (y, then
// x) and returns total, the number of points found. A total larger than
// capacity is a normal result: call again with a larger array, or pass
// capacity 0 (kps may then be NULL) to count only.
//
// threshold is 0..255. Scratch: 3 * cols bytes with nonmax, none without.
int32_t fast_detect(struct KeyPoint *kps, int32_t capacity,
                    const struct Mat *src, int threshold, int nonmax);

// ---- ORB (oriented FAST and rotated BRIEF), single scale, integer only -----
//
// Not bit-compatible with cv2.ORB: the 256 test pairs come from our own seeded
// generator (not OpenCV's learned table), and the orientation is quantized to
// ORB_ANGLE_STEPS steps without atan2. Matching quality under rotation is on
// par with cv2.ORB_create(nlevels=1); see docs/kodak-results.md.

#define ORB_DESC_BYTES  32 // 256-bit descriptor
#define ORB_ANGLE_STEPS 32 // 11.25 degrees per step
#define ORB_BORDER      19 // keypoints keep this distance from every edge

// Detects up to max_kps oriented keypoints in a CV_8UC1 image.
//  1. fast_detect(threshold, non-max suppression);
//  2. drops corners closer than ORB_BORDER to an edge
//     (kept: ORB_BORDER <= x < cols - ORB_BORDER, same for y);
//  3. keeps the 2 * max_kps with the highest FAST score, then ranks those by
//     the Harris measure 25*det(M) - trace(M)^2 (k = 0.04) of the 3x3 Sobel
//     gradients over a 7x7 window and keeps the first max_kps (both steps
//     highest first, ties in raster order; the order cv2.ORB uses);
//  4. sets angle to the step whose direction is closest to the intensity
//     centroid of the radius-15 disc around the point.
// Writes the kept points in rank order and returns their count (<= max_kps).
// response keeps the FAST score. Runs FAST twice: fast_detect to count the
// corners, then the same search again to fill them in.
// Scratch, all allocated together: 3 * cols bytes for the FAST score rows
// plus n * (sizeof(struct KeyPoint) + 16) bytes for the n FAST corners (16 is
// the size of a rank entry on 32- and 64-bit targets; on AVR it is 12, so
// n * 28 in all).
int32_t orb_detect(struct KeyPoint *kps, int32_t max_kps, const struct Mat *src,
                   int threshold);

// Computes one 256-bit rotated BRIEF descriptor per keypoint. smooth is the
// image to sample, CV_8UC1, normally gaussian_blur(src, 5 or 7). Each keypoint
// must lie ORB_BORDER inside smooth and have an angle in 0..ORB_ANGLE_STEPS-1
// (as orb_detect returns). Bit i (byte i / 8, bit i % 8) is 1 when the first
// point of pair i, rotated by the angle, is darker than the second. No
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

// Brute-force matching: for each query descriptor, the train descriptor with
// the smallest Hamming distance (ties: lowest index). With cross_check != 0 a
// pair is kept only when the query is also the best match of its train
// descriptor (ties: lowest index). Writes matches in query order and returns
// their count (<= nq). Scratch: nt * 4 bytes with cross_check, none without.
int32_t match_descriptors(struct DMatch *matches,
                          const uint8_t (*query)[ORB_DESC_BYTES], int32_t nq,
                          const uint8_t (*train)[ORB_DESC_BYTES], int32_t nt,
                          int cross_check);

#endif // FEATURES2D_H_
