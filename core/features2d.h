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

#endif // FEATURES2D_H_
