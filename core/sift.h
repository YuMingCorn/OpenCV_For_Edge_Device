#ifndef SIFT_H_
#define SIFT_H_

// SIFT keypoints and 128-byte descriptors, after cv2.SIFT (OpenCV 4.10).
// Float only: compiled when IMPROC_HAS_FLOAT is 1. Built without fused
// multiply-add and with its own exp, atan2, pow, sin and cos, so the output is
// the same bytes on every target. docs/kodak-results.md compares it with cv2.

#include <stdint.h>

#include "mat.h"

#if IMPROC_HAS_FLOAT

#define SIFT_DESC_BYTES 128 // 4 x 4 cells x 8 orientations

// Positions are in input pixels, centres at integers, as cv2.KeyPoint (with
// upsample, also cv2's 0.25 pixel offset).
struct SiftKeyPoint {
    float   x;
    float   y;
    float   size;     // diameter of the neighbourhood, as cv2
    float   angle;    // degrees in [0, 360), from +x towards +y, as cv2
    float   response; // |DoG| at the refined extremum, image scaled to 0..1
    int32_t octave;   // -1 for the 2x upsampled octave, then 0, 1, ...
    int32_t layer;    // 1 .. n_octave_layers (sift_compute: 0 .. n + 2)
};

struct SiftParams {
    int    n_octave_layers;    // 1 .. 15 (cv2: 3)
    float  contrast_threshold; // 0 .. 100 (cv2: 0.04)
    float  edge_threshold;     // (0, 1e4] (cv2: 10)
    double sigma;              // 0.1 .. 32 (cv2: 1.6), double as cv2
    int    upsample;           // != 0: start from the 2x image, as cv2
};

// Fills p with the cv2.SIFT_create defaults (upsample on).
void sift_default_params(struct SiftParams *p);

// Scratch bytes of one call for a rows x cols image (p NULL: defaults); 0
// when there is no octave. About 34 MB for 640 x 480 with the defaults.
size_t sift_scratch_bytes(int rows, int cols, const struct SiftParams *p);

// Keypoints of a CV_8UC1 image, as cv2 SIFT.detect with nfeatures =
// capacity: copies dropped, then the capacity strongest kept, written in
// decreasing response (ties: octave, layer, y, x, angle ascending). Returns
// the count. cv2 also keeps keypoints tied with the last one kept, so it may
// return more than capacity; ours are then a subset. p NULL for the defaults.
int32_t sift_detect(struct SiftKeyPoint *kps, int32_t capacity,
                    const struct Mat *src, const struct SiftParams *p);

// Descriptors of n given keypoints, as cv2 SIFT.compute: row i for kps[i].
// The 2x image is used exactly when some keypoint has octave -1, whatever
// p->upsample says; the scratch is sift_scratch_bytes with that upsample.
void sift_compute(uint8_t (*desc)[SIFT_DESC_BYTES],
                  const struct SiftKeyPoint *kps, int32_t n,
                  const struct Mat *src, const struct SiftParams *p);

// sift_detect, then the descriptors from the same scale space (desc may be
// NULL), as cv2 detectAndCompute.
int32_t sift_detect_compute(struct SiftKeyPoint *kps,
                            uint8_t (*desc)[SIFT_DESC_BYTES], int32_t capacity,
                            const struct Mat *src, const struct SiftParams *p);

struct SiftMatch {
    int32_t query;    // index into the query descriptors
    int32_t train;    // index into the train descriptors
    int32_t distance; // squared L2 distance (0 .. 128 * 255 * 255)
};

// Brute-force L2 matching with Lowe's ratio test: the best train descriptor
// (squared distance d1, runner-up d2 over all others) is kept when
// d1 < ratio^2 * d2, or always when nt == 1. ratio in (0, 1]. Writes matches
// in query order and returns their count.
int32_t sift_match(struct SiftMatch *matches,
                   const uint8_t (*query)[SIFT_DESC_BYTES], int32_t nq,
                   const uint8_t (*train)[SIFT_DESC_BYTES], int32_t nt,
                   float ratio);

#endif // IMPROC_HAS_FLOAT

#endif // SIFT_H_
