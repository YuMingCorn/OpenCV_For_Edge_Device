#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "mat.h"

enum FlipCodes { FLIP_VERTICAL = 0, FLIP_HORIZONTAL = 1, FLIP_BOTH = -1 };

// Flips src into dst (same type and size, CV_8UC1/3/4). FLIP_VERTICAL reverses
// row order, FLIP_HORIZONTAL reverses pixel order within each row.
void flip(struct Mat *dst, const struct Mat *src, int flip_code);

// Copies the region of src whose top-left pixel is (x, y) into dst. The region
// size is the size of dst, and it must lie inside src. Works for any type;
// src and dst must have the same type and distinct buffers. Use it to cut a
// region of interest out of a camera frame before further processing.
void crop(struct Mat *dst, const struct Mat *src, int x, int y);

// INTER_NEAREST matches OpenCV's INTER_NEAREST_EXACT (source index
// floor((d + 0.5) * src / dst)), not its INTER_NEAREST (floor(d * src / dst)).
enum InterpolationFlags { INTER_NEAREST, INTER_LINEAR, INTER_AREA };

// Resizes src into dst (same type, CV_8UC1/3/4). The target size is the size
// of dst. Integer arithmetic only, following OpenCV's 8-bit definitions.
//
// INTER_LINEAR uses half-pixel centres. As in OpenCV, the source position is
// rounded to float32 first (emulated with integers), and the two Q11 weights
// come from its fractional part, each rounded on its own. x is clamped at the
// edges but y is not (the row indices are clamped instead), and the vertical
// step rounds as OpenCV's vector code does.
//
// INTER_AREA, when neither axis grows, averages the exact source area covered
// by each destination pixel. Per axis, destination index d covers
// [d*s_len, (d+1)*s_len) and source index k covers [k*d_len, (k+1)*d_len),
// both in units of 1/d_len. A weight is the integer overlap of the two
// intervals, so the weights of one destination pixel sum to s_len. The mean
// (divided by den = src_cols*src_rows) is rounded half to even, except for an
// exact 2x2 reduction, which rounds half up, as in OpenCV. When either axis
// grows, INTER_AREA runs the linear kernel with OpenCV's area mapping instead.
// The averaging path requires src rows*cols <= 2^24 so the sums fit in
// uint32_t, and uses 2*dst_cols*channels*4 bytes of scratch from fast_malloc.
void resize(struct Mat *dst, const struct Mat *src, int interpolation);

#endif // GEOMETRY_H_
