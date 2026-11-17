#ifndef DRAW_H_
#define DRAW_H_

#include <stdint.h>

#include "mat.h"

// Drawing helpers for visualising results (detections, boxes, labels).
//
// Common rules for every function:
// - img is CV_8UC1, CV_8UC3 or CV_8UC4 and is drawn in place.
// - color holds one value per channel in the Mat's own channel order (no BGR
//   assumption); only the first cn entries are read.
// - x is the column and y the row. Coordinates are int32_t and may lie
//   anywhere in the int32_t range: whatever falls outside the image is
//   clipped silently. Shapes partly or fully outside the image are normal use.
// - ASSERT only on misuse: NULL img, color or text, an unsupported type, an
//   invalid thickness, scale or radius (limits below).
// - Integers only (no float), no scratch memory, no stdio. Run time depends on
//   the visible part of the shape (and, for thick lines, on the thickness),
//   not on the size of its coordinates: lines are clipped before they are
//   walked and circles are scanned over the image rows and columns they can
//   reach. Image sides are assumed below INT32_MAX - 2^16.

// Largest thickness for draw_line and draw_circle. A thick line stamps its
// brush at every step, so the work grows with thickness^2 times the length;
// 255 keeps the worst case (an image-wide line) well under a second. OpenCV
// allows 32767. For a larger dot or disc use a filled circle.
#define DRAW_MAX_THICKNESS 255

// A single pixel. Nothing happens when (x, y) is outside the image. For a
// larger dot use draw_circle with a negative thickness.
void draw_point(struct Mat *img, int32_t x, int32_t y, const uchar color[4]);

// Line from (x0, y0) to (x1, y1), both ends included. 1 <= thickness <=
// DRAW_MAX_THICKNESS.
//
// thickness 1: the 8-connected Bresenham line of cv2.line(LINE_8). The line is
// first clipped to the image like cv2.clipLine (integer arithmetic, quotients
// truncated toward zero), then walked from the left end. The result is byte
// for byte the same as cv2.line(img, p0, p1, color, 1, LINE_8) whenever every
// coordinate is within +-2^26. Beyond that cv2 clips with double and may round
// one end differently; this code stays exact.
//
// thickness t >= 2: a round brush of diameter t is stamped at every pixel of
// that thickness-1 line (clipped to the image grown by the brush radius). The
// brush holds the offsets (i, j) with (2i - e)^2 + (2j - e)^2 <= t^2, where
// e = 1 for even t and 0 for odd t, so an even brush is centred half a pixel
// right of and below the line. The line is t pixels wide with round ends.
// Differs from cv2.line, which draws a filled polygon plus end circles about
// 2 * ceil(t / 2) + 1 pixels wide.
void draw_line(struct Mat *img, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
               int thickness, const uchar color[4]);

// Circle with centre (cx, cy) and radius >= 0. thickness is negative (filled)
// or 1..DRAW_MAX_THICKNESS; 0 is misuse.
//
// thickness < 0: every pixel with dx^2 + dy^2 <= r^2. Byte for byte the same
// as cv2.circle(img, c, r, color, -1, LINE_8).
// thickness 1: for each k with 2k^2 <= r^2 and g = floor(sqrt(r^2 - k^2)), the
// eight points (+-g, +-k) and (+-k, +-g). Byte for byte the same as
// cv2.circle(img, c, r, color, 1, LINE_8).
// thickness t >= 2: the ring between the radii ri + 1/2 and ro + 1/2, where
// ro = r + t / 2 and ri = ro - t: the pixels with
// ri^2 + ri < dx^2 + dy^2 <= ro^2 + ro (no hole when ri < 0). It is t pixels
// wide around radius r (around r + 1/2 for even t, like the brush of
// draw_line), and the half-pixel radii avoid single-pixel bumps at the four
// ends of the axes. Differs from cv2.circle, which draws a polyline about
// 2 * ceil(t / 2) + 1 pixels wide.
// Radius 0 with thickness 1 or < 0 is the centre pixel alone.
void draw_circle(struct Mat *img, int32_t cx, int32_t cy, int32_t radius,
                 int thickness, const uchar color[4]);

// Built-in 5x8 bitmap font. Each character takes a cell of DRAW_FONT_W x
// DRAW_FONT_H pixels times scale: the glyph is 5 columns wide plus one blank
// spacing column; rows 0..6 hold capitals and digits (baseline on row 6) and
// row 7 holds descenders (g, j, p, q, y, comma, semicolon, underscore).
#define DRAW_FONT_W         6
#define DRAW_FONT_H         8
#define DRAW_TEXT_MAX_SCALE 1024

// Draws a NUL-terminated byte string on one line. (x, y) is the top-left
// corner of the text box: character i covers columns
// x + i * 6 * scale .. x + (i + 1) * 6 * scale - 1 (the last of them blank)
// and rows y .. y + 8 * scale - 1. Each font pixel becomes a scale x scale
// block. 1 <= scale <= DRAW_TEXT_MAX_SCALE. Only ink pixels are written; the
// background is left unchanged.
//
// Bytes:
// - 0x20..0x7E (printable ASCII): its glyph.
// - 0x80..0xBF (UTF-8 continuation bytes): nothing drawn and no space taken,
//   so a well-formed UTF-8 character (e.g. Chinese) takes one cell.
// - every other byte (control bytes such as '\n' and '\t', 0x7F, UTF-8 lead
//   bytes 0xC0..0xFF): a hollow box glyph in one cell. There is no line break.
void draw_text(struct Mat *img, const char *text, int32_t x, int32_t y,
               int scale, const uchar color[4]);

// Width in pixels of the text box that draw_text uses for text and scale:
// cells * 6 * scale, where cells counts every byte except 0x80..0xBF. The
// height is DRAW_FONT_H * scale. ASSERTs that the width fits int32_t.
int32_t draw_text_width(const char *text, int scale);

#endif // DRAW_H_
