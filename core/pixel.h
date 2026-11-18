#ifndef PIXEL_H_
#define PIXEL_H_

#include <stdint.h>

#include "mat.h"
#include "utils.h"

enum ThresholdTypes {
    THRESH_BINARY,
    THRESH_BINARY_INV,
    THRESH_TRUNC,
    THRESH_TOZERO,
    THRESH_TOZERO_INV
};

// src and dst: CV_8UC1/3/4, same type and size. Every channel is thresholded.
void threshold(struct Mat *dst, const struct Mat *src, uchar thresh,
               uchar maxval, int type);

// dst[i] = lut[src[i]] for every channel. Same type/size rules as threshold.
void mat_lut(struct Mat *dst, const struct Mat *src, const uchar lut[256]);

// dst = |a - b| for every channel, the per-pixel difference used for frame
// differencing (motion detection). a, b, dst: CV_8UC1/3/4, same type and size;
// dst must not share a buffer with a or b. a and b may be the same image.
void absdiff(struct Mat *dst, const struct Mat *a, const struct Mat *b);

// dst = saturate_u8(round(src * alpha_q8 / 256 + beta)), CV_8UCn -> CV_8UCn,
// rounded half to even like OpenCV. alpha_q8 in [-32768, 32767], beta in
// [-65536, 65535]. Intermediates are int32_t: 255 * 32767 + 65535 * 256 < 2^25.
void convert_scale(struct Mat *dst, const struct Mat *src, int32_t alpha_q8,
                   int32_t beta);

// dst = saturate_u8(round(|src * alpha_q8 / 256 + beta|)), CV_16SCn -> CV_8UCn,
// rounded half to even. Same ranges. 32768 * 32768 + 65536 * 256 < 2^31, so int32_t does not
// overflow.
void convert_scale_abs(struct Mat *dst, const struct Mat *src, int32_t alpha_q8,
                       int32_t beta);

#endif // PIXEL_H_
