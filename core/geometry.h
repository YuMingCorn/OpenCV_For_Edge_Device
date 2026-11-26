#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "mat.h"

enum FlipCodes { FLIP_VERTICAL = 0, FLIP_HORIZONTAL = 1, FLIP_BOTH = -1 };

// Flips src into dst (same type and size, CV_8UC1/3/4). FLIP_VERTICAL reverses
// row order, FLIP_HORIZONTAL reverses pixel order within each row.
void flip(struct Mat *dst, const struct Mat *src, int flip_code);

// Copies the dst-sized region of src with top-left pixel (x, y) into dst. Any
// type; same type, distinct buffers, region inside src.
void crop(struct Mat *dst, const struct Mat *src, int x, int y);

// INTER_NEAREST matches OpenCV's INTER_NEAREST_EXACT (source index
// floor((d + 0.5) * src / dst)), not its INTER_NEAREST (floor(d * src / dst)).
enum InterpolationFlags { INTER_NEAREST, INTER_LINEAR, INTER_AREA };

// Resizes src to the size of dst (same type, CV_8UC1/3/4) with integer
// arithmetic, following OpenCV's 8-bit definitions.
//
// INTER_LINEAR: half-pixel centres; the source position is rounded to float32
// (emulated) and each Q11 weight rounded on its own, as in OpenCV.
//
// INTER_AREA when neither axis grows: the exact area mean, rounded half to
// even (half up for an exact 2x2 reduction, as in OpenCV). Needs src
// rows*cols <= 2^24 and 2*dst_cols*channels*4 bytes of scratch. When an axis
// grows it runs the linear kernel with OpenCV's area mapping.
void resize(struct Mat *dst, const struct Mat *src, int interpolation);

#endif // GEOMETRY_H_
