#ifndef SIFT_H_
#define SIFT_H_

// SIFT (scale-invariant feature transform) keypoints and 128-byte descriptors,
// after Lowe 2004 and the layout of cv2.SIFT. Float only: compiled when
// IMPROC_HAS_FLOAT is 1.
//
// Not bit-compatible with cv2.SIFT. It uses its own exp, atan2, sin and cos
// polynomials and is built without fused multiply-add, so the output is the
// same bytes on every target (x86-64, aarch64, ...). Keypoint positions,
// scales and matching quality are compared with cv2.SIFT in
// docs/kodak-results.md.

#include <stdint.h>

#include "mat.h"

#if IMPROC_HAS_FLOAT

#define SIFT_DESC_BYTES 128 // 4 x 4 cells x 8 orientations

// One keypoint. Coordinates are in pixels of the input image, with pixel
// centres at integers (x is the column, y the row), as cv2.KeyPoint. With
// upsample, positions keep cv2's convention of half the upsampled position,
// which is 0.25 pixel right of and below the true position (upsampled pixel
// u lies at u / 2 - 0.25 of the input).
struct SiftKeyPoint {
    float   x;
    float   y;
    float   size;     // diameter of the meaningful neighbourhood, as cv2
    float   angle;    // degrees in [0, 360); 0 is +x, and angles grow from
                      // +x towards +y (clockwise on screen), as cv2
    float   response; // |DoG| at the refined extremum, image scaled to 0..1
    int32_t octave;   // -1 for the 2x upsampled octave, then 0, 1, ...
    int32_t layer;    // 1 .. n_octave_layers
};

struct SiftParams {
    int   n_octave_layers;    // scales per octave, s (cv2: 3)
    float contrast_threshold; // reject |DoG| below this / s (cv2: 0.04)
    float edge_threshold;     // max principal curvature ratio (cv2: 10)
    float sigma;              // blur of octave 0, layer 0 (cv2: 1.6)
    int   upsample;           // != 0: start from the 2x upsampled image, as
                              // cv2 (finds more small features, 4x memory)
};

// Fills p with the cv2.SIFT_create defaults (upsample on).
void sift_default_params(struct SiftParams *p);

// Scratch bytes sift_detect_compute allocates for a rows x cols image
// (rows, cols >= 0; p NULL for the defaults). Octave 0 has R x C pixels:
// R = 2 * rows and C = 2 * cols with upsample, else rows and cols. The
// scratch is (n_octave_layers + 4) * R * C + 360 + k floats: the Gaussian
// images of one octave, one blur buffer, the descriptor histogram and k
// kernel weights (k = 14 for the defaults). Returns 0 when there is no
// octave (min(R, C) < 6). ASSERT fails when the size does not fit size_t.
// Example: a 640 x 480 image with the defaults needs about 34 MB; without
// upsample, 8.6 MB.
size_t sift_scratch_bytes(int rows, int cols, const struct SiftParams *p);

// Detects SIFT keypoints in a CV_8UC1 image and computes their descriptors.
//
// Steps: optional 2x bilinear upsample; Gaussian scale space with
// n_octave_layers + 3 images per octave (an octave halves the size, the
// first image of an octave is every second pixel of image n_octave_layers of
// the octave before); difference-of-Gaussian extrema over 26 neighbours,
// at least 5 pixels from the border; sub-pixel refinement (at most 5 steps);
// contrast and edge rejection; one keypoint per orientation peak (36-bin
// histogram, peaks >= 80 % of the highest); a 4 x 4 x 8 descriptor, each
// value clipped at 0.2 of the norm and scaled by 512 to a byte.
// The octave count is round(log2(min(R, C))) - 2, as cv2; an octave halves
// the size of the one before (rounding down).
//
// When more than capacity keypoints are found, the capacity keypoints with
// the highest response are kept (cv2 nfeatures). They are written to kps and,
// when desc is not NULL, desc (row i is the descriptor of kps[i]) in
// decreasing response; equal responses are ordered by octave, layer, y, x and
// angle, all ascending. Two start pixels can refine to the same keypoint: a
// keypoint equal to the one before it (same x, y, size and angle) is dropped,
// as cv2 does. Returns the count written (<= capacity; it can be below
// capacity after dropping copies even when more keypoints exist). capacity 0
// returns 0 without any work.
//
// p may be NULL for the defaults. Processes one octave at a time, so the
// scratch is sift_scratch_bytes(), allocated once. With desc NULL only the
// keypoints are found (the scratch is the same).
int32_t sift_detect_compute(struct SiftKeyPoint *kps,
                            uint8_t (*desc)[SIFT_DESC_BYTES], int32_t capacity,
                            const struct Mat *src, const struct SiftParams *p);

struct SiftMatch {
    int32_t query;    // index into the query descriptors
    int32_t train;    // index into the train descriptors
    int32_t distance; // squared L2 distance (0 .. 128 * 255 * 255)
};

// Brute-force L2 matching with Lowe's ratio test. For each query descriptor,
// finds the train descriptor with the smallest squared distance d1 and the
// second smallest d2 (counting every train descriptor, so two equally close
// ones give d1 == d2). The pair is kept when d1 < ratio^2 * d2 (distance
// ratio below ratio), so a tie for the best is never kept; with nt == 1
// every best match is kept. ratio in (0, 1]; Lowe uses 0.8, and ratio 1
// keeps every match strictly better than the runner-up. Writes matches in
// query order and returns their count (<= nq). No scratch.
int32_t sift_match(struct SiftMatch *matches,
                   const uint8_t (*query)[SIFT_DESC_BYTES], int32_t nq,
                   const uint8_t (*train)[SIFT_DESC_BYTES], int32_t nt,
                   float ratio);

#endif // IMPROC_HAS_FLOAT

#endif // SIFT_H_
