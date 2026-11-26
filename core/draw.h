#ifndef DRAW_H_
#define DRAW_H_

#include <stdint.h>

#include "mat.h"

// All functions draw in place on a CV_8UC1/3/4 img. color holds one value per
// channel in the Mat's own order; only the first cn entries are read. x is the
// column, y the row; any int32_t coordinate is allowed and the part outside
// the image is clipped. ASSERT on NULL arguments, an unsupported type, or a
// thickness, scale or radius out of range. Integers only, no scratch. Run time
// depends on the visible part of the shape (and on thickness), not on its
// coordinates. Image sides must be below INT32_MAX - 2^16.

// Work grows with thickness^2 times the length (OpenCV allows 32767).
#define DRAW_MAX_THICKNESS 255

// A single pixel; nothing happens outside the image.
void draw_point(struct Mat *img, int32_t x, int32_t y, const uchar color[4]);

// Line from (x0, y0) to (x1, y1), both ends included. 1 <= thickness <=
// DRAW_MAX_THICKNESS.
//
// thickness 1: byte for byte cv2.line(img, p0, p1, color, 1, LINE_8) when
// every coordinate is within +-2^26 (beyond that cv2 clips in double and may
// round an end differently).
//
// thickness t >= 2: a round brush with offsets (i, j),
// (2i - e)^2 + (2j - e)^2 <= t^2 (e = 1 for even t, else 0), stamped on every
// pixel of the thickness-1 line: t pixels wide with round ends, centred half a
// pixel right of and below the line for even t. cv2.line draws about
// 2 * ceil(t / 2) + 1 pixels wide instead.
void draw_line(struct Mat *img, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
               int thickness, const uchar color[4]);

// Circle with centre (cx, cy) and radius >= 0. thickness is negative (filled)
// or 1..DRAW_MAX_THICKNESS; 0 is misuse.
//
// thickness < 0 (pixels with dx^2 + dy^2 <= r^2) and thickness 1 are byte for
// byte cv2.circle(img, c, r, color, thickness, LINE_8); radius 0 then draws
// the centre pixel alone.
// thickness t >= 2: the ring ri^2 + ri < dx^2 + dy^2 <= ro^2 + ro with
// ro = r + t / 2 and ri = ro - t (no hole when ri < 0), t pixels wide around
// r (around r + 1/2 for even t).
// cv2.circle draws about 2 * ceil(t / 2) + 1 pixels wide instead.
void draw_circle(struct Mat *img, int32_t cx, int32_t cy, int32_t radius,
                 int thickness, const uchar color[4]);

// Built-in font: a cell of DRAW_FONT_W x DRAW_FONT_H pixels per character (a
// 5-column glyph plus one blank column; baseline on row 6, descenders on 7).
#define DRAW_FONT_W         6
#define DRAW_FONT_H         8
#define DRAW_TEXT_MAX_SCALE 1024

// Draws a NUL-terminated string on one line with its top-left corner at
// (x, y); each font pixel is a scale x scale block, 1 <= scale <=
// DRAW_TEXT_MAX_SCALE. Only ink pixels are written. Bytes 0x20..0x7E draw
// their glyph, 0x80..0xBF (UTF-8 continuation) draw nothing and take no cell,
// and every other byte, '\n' included, draws a hollow box in one cell.
void draw_text(struct Mat *img, const char *text, int32_t x, int32_t y,
               int scale, const uchar color[4]);

// Width of the draw_text box: cells * DRAW_FONT_W * scale, with cells as in
// draw_text (the height is DRAW_FONT_H * scale). Same text and scale rules as
// draw_text; also ASSERTs that the width fits int32_t.
int32_t draw_text_width(const char *text, int scale);

#endif // DRAW_H_
