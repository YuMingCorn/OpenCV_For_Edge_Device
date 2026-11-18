#include "improc.h"
#include "kodak.h"

#include <string.h>

// Drawing on one Kodak image. Every op draws on a copy of the RGB image; the
// shapes derive from the image size (768x512 or 512x768) and include parts
// outside the image, end points far outside, and +-2^24 or 2^20 coordinates.
// tools/opencv_ref/draw.py repeats the same shapes with cv2.line and
// cv2.circle (draw_lines, draw_circles) or a numpy model of the thick rules
// (draw_thick). draw_text has no reference and is checked by its hash only.

#define MARGIN 37 // fan end points lie this far outside the image
#define STEP   53 // spacing of the fan end points
#define BIG    (1 << 24)
#define HUGE_R (1 << 20)

static struct Mat *copy_of(const struct Mat *rgb) {
    struct Mat *out = alloc_mat(rgb->rows, rgb->cols, CV_8UC3);
    memcpy(out->data, rgb->data, (size_t)rgb->rows * rgb->step[0]);
    return out;
}

// Colour number i: (i * 67 + 20, i * 29 + 80, 255 - i * 13), each mod 256.
static const uchar *color_of(int i, uchar c[4]) {
    c[0] = (uchar)(i * 67 + 20);
    c[1] = (uchar)(i * 29 + 80);
    c[2] = (uchar)(255 - i * 13);
    c[3] = 255;
    return c;
}

// Thickness-1 lines: a fan from the centre to points around the image, then
// lines fully outside, short lines, a single point and huge coordinates.
static struct Mat *op_lines(const struct Mat *rgb) {
    struct Mat *out = copy_of(rgb);
    const int   w = rgb->cols, h = rgb->rows, cx = w / 2, cy = h / 2;
    uchar       c[4];
    int         i = 0;
    for (int x = -MARGIN; x <= w + MARGIN; x += STEP) {
        draw_line(out, cx, cy, x, -MARGIN, 1, color_of(i++, c));
        draw_line(out, cx, cy, x, h - 1 + MARGIN, 1, color_of(i++, c));
    }
    for (int y = -MARGIN; y <= h + MARGIN; y += STEP) {
        draw_line(out, cx, cy, -MARGIN, y, 1, color_of(i++, c));
        draw_line(out, cx, cy, w - 1 + MARGIN, y, 1, color_of(i++, c));
    }
    draw_line(out, -50, -10, -5, h + 100, 1, color_of(i++, c)); // outside
    draw_line(out, 2, 2, 9, 5, 1, color_of(i++, c));
    draw_line(out, 5, 9, 5, 9, 1, color_of(i++, c)); // one pixel
    draw_line(out, w - 1, h - 1, w - 30, h - 4, 1, color_of(i++, c));
    draw_line(out, -BIG, h / 3, BIG, h / 3 + 5, 1, color_of(i++, c));
    draw_line(out, w / 3, -BIG, w / 3 + 9, BIG, 1, color_of(i++, c));
    draw_line(out, -BIG, -BIG + 3, BIG, BIG, 1, color_of(i++, c));
    draw_line(out, BIG, -BIG, -BIG + 11, BIG, 1, color_of(i++, c));
    return out;
}

// Thickness-1 and filled circles: concentric rings that end outside the
// image, filled discs over the corners, rings around a corner, and two
// circles with a 2^20 radius whose edge crosses the image.
static struct Mat *op_circles(const struct Mat *rgb) {
    struct Mat *out = copy_of(rgb);
    const int   w = rgb->cols, h = rgb->rows, cx = w / 2, cy = h / 2;
    const int   rmax = (w < h ? w : h) / 2 + 40;
    uchar       c[4];
    int         i = 0;
    draw_circle(out, 0, 0, 45, -1, color_of(i++, c));
    draw_circle(out, w - 1, h - 1, 60, -1, color_of(i++, c));
    draw_circle(out, w + 10, cy, 30, -1, color_of(i++, c));
    draw_circle(out, -100, -100, 20, -1, color_of(i++, c)); // outside
    draw_circle(out, cx, -HUGE_R, HUGE_R + 30, -1, color_of(i++, c));
    for (int r = 0; r <= rmax; r += 9) {
        draw_circle(out, cx, cy, r, 1, color_of(i++, c));
    }
    for (int r = 10; r <= 70; r += 15) {
        draw_circle(out, w - 20, 15, r, 1, color_of(i++, c));
    }
    draw_circle(out, cx, cy + HUGE_R, HUGE_R - h / 4, 1, color_of(i++, c));
    draw_circle(out, 7, 7, 3, -1, color_of(i++, c));
    return out;
}

// Thick lines (thickness 2..9, one of 25 and a thick dot) and rings
// (thickness 2..6 and 7 over a corner).
static struct Mat *op_thick(const struct Mat *rgb) {
    struct Mat *out = copy_of(rgb);
    const int   w = rgb->cols, h = rgb->rows, cx = w / 2, cy = h / 2;
    uchar       c[4];
    int         i = 0;
    for (int k = 0; k < 8; k++) {
        const int t = k + 2;
        draw_line(out, w * k / 8, -20, w - 1 - w * k / 8 + 15, h + 20, t,
                  color_of(i++, c));
        draw_line(out, -30, h * k / 8 + 10, w + 30, h * k / 8 + 40, t,
                  color_of(i++, c));
    }
    for (int k = 0; k < 5; k++) {
        draw_circle(out, cx, cy, 30 + 25 * k, k + 2, color_of(i++, c));
    }
    draw_circle(out, w - 1, 0, 80, 7, color_of(i++, c));
    draw_line(out, 20, h - 30, w - 40, 30, 25, color_of(i++, c));
    draw_line(out, 40, 40, 40, 40, 15, color_of(i++, c));
    draw_line(out, -BIG, cy, BIG, cy + 3, 4, color_of(i++, c));
    return out;
}

// Every printable character at scale 1 and 2 (clipped at the right edge on
// the portrait images), a scale 3 label, text crossing the left and bottom
// edges, and UTF-8 bytes (one box per character, '\n' is a box).
static struct Mat *op_text(const struct Mat *rgb) {
    struct Mat *out = copy_of(rgb);
    const int   h   = rgb->rows;
    char        ascii[96];
    for (int k = 0; k < 95; k++) {
        ascii[k] = (char)(0x20 + k);
    }
    ascii[95] = 0;
    uchar c[4];
    draw_text(out, ascii, 4, 4, 1, color_of(0, c));
    draw_text(out, ascii, 4, 16, 2, color_of(1, c));
    draw_text(out, "Kodak draw_text 3x", 4, 40, 3, color_of(2, c));
    draw_text(out, "left edge", -10, 70, 2, color_of(3, c));
    draw_text(out, "bottom edge", 30, h - 5, 2, color_of(4, c));
    draw_text(out, "\xE4\xB8\xAD\xE6\x96\x87 OK\n", 4, 90, 2, color_of(5, c));
    return out;
}

const struct KodakOp kodak_ops[] = {
    {"draw_lines", op_lines},
    {"draw_circles", op_circles},
    {"draw_thick", op_thick},
    {"draw_text", op_text},
};
const int kodak_op_count = (int)(sizeof(kodak_ops) / sizeof(kodak_ops[0]));
