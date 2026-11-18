#ifndef COLOR_H_
#define COLOR_H_

#include "mat.h"
#include "utils.h"

enum ColorConversionCodes {
    COLOR_RGB2GRAY,
    COLOR_BGR2GRAY,
    COLOR_RGBA2GRAY,
    COLOR_BGRA2GRAY,
    COLOR_GRAY2RGB,
    COLOR_GRAY2BGR = COLOR_GRAY2RGB,
    COLOR_GRAY2RGBA,
    COLOR_GRAY2BGRA = COLOR_GRAY2RGBA,
    COLOR_RGB2BGR,
    COLOR_BGR2RGB = COLOR_RGB2BGR,
    COLOR_RGBA2BGRA,
    COLOR_BGRA2RGBA = COLOR_RGBA2BGRA,
    COLOR_RGB2RGBA,
    COLOR_BGR2BGRA = COLOR_RGB2RGBA,
    COLOR_RGBA2RGB,
    COLOR_BGRA2BGR = COLOR_RGBA2RGB,
    COLOR_RGB2BGRA,
    COLOR_BGR2RGBA = COLOR_RGB2BGRA,
    COLOR_RGBA2BGR,
    COLOR_BGRA2RGB = COLOR_RGBA2BGR,
    // CV_8UC3 to CV_8UC3. HSV uses H 0..179, S and V 0..255.
    COLOR_RGB2HSV,
    COLOR_BGR2HSV,
    COLOR_HSV2RGB,
    COLOR_HSV2BGR,
    // CV_8UC3 to CV_8UC3. OpenCV's YCrCb in the order Y, U (Cb), V (Cr).
    COLOR_RGB2YUV,
    COLOR_BGR2YUV,
    COLOR_YUV2RGB,
    COLOR_YUV2BGR,
    // Camera formats, CV_8UC2 source. A 565 pixel is the 16-bit word
    // RRRRRGGG GGGBBBBB (blue in the low bits, as in OpenCV's BGR565). The
    // plain codes read the word little-endian, like OpenCV; the BE codes read
    // it big-endian (high byte first), which camera modules such as the
    // OV2640 commonly use (not yet confirmed on hardware for this project).
    // Channels expand by shifting left with no bit replication.
    COLOR_BGR5652BGR,
    COLOR_BGR5652RGB,
    COLOR_BGR5652GRAY,
    COLOR_BGR565BE2BGR,
    COLOR_BGR565BE2RGB,
    COLOR_BGR565BE2GRAY,
    // YUV 4:2:2 packed as Y0 U Y1 V, CV_8UC2 source with an even width. Uses
    // limited-range BT.601 (Y 16..235) with OpenCV's integer constants, so it
    // is not the full-range COLOR_YUV2RGB above. GRAY copies Y.
    COLOR_YUV2RGB_YUYV,
    COLOR_YUV2BGR_YUYV,
    COLOR_YUV2GRAY_YUYV
};

// 8-bit color conversion. The caller allocates dst with the matching type and
// the same rows and cols as src. Every code matches OpenCV 4.10 byte for byte,
// except HSV2RGB in OpenCV's scalar row tail, described below.
// Gray uses OpenCV's Q15 weights, Y = (9798R + 19235G + 3735B + 16384) >> 15,
// also for 565 to GRAY. YUV is YCrCb with the two chroma channels swapped.
// HSV2RGB follows OpenCV's vector loop (float32, truncating); OpenCV's scalar
// loop, used for the last width % 32 pixels of a row on AVX2 builds, rounds
// instead and differs by one level on about a third of those pixels.
void cvt_color(struct Mat *dst, const struct Mat *src, int code);

#endif // COLOR_H_
