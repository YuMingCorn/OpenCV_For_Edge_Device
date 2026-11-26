#include "draw.h"

#include "utils.h"

// ---- Common helpers ---------------------------------------------------------

// ASSERTs the arguments every function shares and returns the channel count.
static int draw_channels(const struct Mat *img, const uchar *color) {
    ASSERT(img != NULL && color != NULL);
    const size_t t = mat_type(img);
    ASSERT(t == (size_t)CV_8UC1 || t == (size_t)CV_8UC3 ||
           t == (size_t)CV_8UC4);
    return (int)mat_channel_num(img);
}

static int draw_is_empty(const struct Mat *img) {
    return img->rows <= 0 || img->cols <= 0;
}

// Writes one pixel; the caller has checked that (x, y) is inside the image.
static void put_px(struct Mat *img, int x, int y, const uchar *color, int cn) {
    uchar *p = mat_row(img, y) + (size_t)x * (size_t)cn;
    for (int c = 0; c < cn; c++) {
        p[c] = color[c];
    }
}

// Writes one pixel when (x, y) is inside the image.
static void put_clipped(struct Mat *img, int64_t x, int64_t y,
                        const uchar *color, int cn) {
    if (x >= 0 && y >= 0 && x < img->cols && y < img->rows) {
        put_px(img, (int)x, (int)y, color, cn);
    }
}

// Fills columns x0..x1 of row y, clipped to the image. 32-bit arguments keep
// the code small on 8-bit targets; callers with 64-bit values clamp first.
static void hline_clipped(struct Mat *img, int32_t y, int32_t x0, int32_t x1,
                          const uchar *color, int cn) {
    if (y < 0 || y >= img->rows) {
        return;
    }
    if (x0 < 0) {
        x0 = 0;
    }
    if (x1 > img->cols - 1) {
        x1 = img->cols - 1;
    }
    for (int32_t x = x0; x <= x1; x++) {
        put_px(img, (int)x, (int)y, color, cn);
    }
}

// v clamped to -1..cols, which hline_clipped treats like v.
static int32_t clamp_col(int64_t v, int cols) {
    return v < -1 ? -1 : v > cols ? (int32_t)cols : (int32_t)v;
}

// floor(sqrt(n)), bit by bit.
static uint64_t isqrt_u64(uint64_t n) {
    uint64_t res = 0;
    uint64_t bit = (uint64_t)1 << 62;
    while (bit > n) {
        bit >>= 2;
    }
    while (bit != 0) {
        if (n >= res + bit) {
            n -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

static uint64_t abs_u64(int64_t v) {
    return v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
}

void draw_point(struct Mat *img, int32_t x, int32_t y, const uchar color[4]) {
    const int cn = draw_channels(img, color);
    put_clipped(img, x, y, color, cn);
}

// ---- Lines ------------------------------------------------------------------

// a * b / c truncated toward zero. |a| and |b| are below 2^32 and c != 0, so
// the product of the magnitudes fits uint64_t, and so does the result.
static int64_t mul_div_trunc(int64_t a, int64_t b, int64_t c) {
    const uint64_t q   = abs_u64(a) * abs_u64(b) / abs_u64(c);
    const int      neg = ((a < 0) != (b < 0)) != (c < 0);
    return neg ? -(int64_t)q : (int64_t)q;
}

static int out_code(int64_t x, int64_t y, int64_t right, int64_t bottom) {
    return (x < 0) + (x > right) * 2 + (y < 0) * 4 + (y > bottom) * 8;
}

// Moves the end (*u, *v) along the line to the other end (uo, vo) until
// v == a: *u += (a - *v) * (*u - uo) / (*v - vo). The same quotient as
// cv2.clipLine's (a - v1) * (u2 - u1) / (v2 - v1), whose numerator and
// denominator differ from these only in sign.
static void clip_end(int64_t *u, int64_t *v, int64_t uo, int64_t vo,
                     int64_t a) {
    *u += mul_div_trunc(a - *v, *u - uo, *v - vo);
    *v = a;
}

// cv2.clipLine on the rectangle [0, w) x [0, h): the same steps in the same
// order, with exact integer quotients instead of double. p is x1, y1, x2, y2.
// Returns 1 when part of the line is inside; p then holds the clipped ends.
// Every intermediate end lies between the two input ends, so the differences
// passed to mul_div_trunc stay below 2^32 in magnitude.
//
// cv2 runs two stages while the ends are neither both inside nor both beyond
// the same edge: stage 0 moves each end that is above or below the image onto
// the top or bottom row (end 1 first), stage 1 moves each end still left or
// right of the image onto the first or last column. Written as one loop so
// the 64-bit arithmetic appears once (it is costly on 8-bit targets).
static int clip_line(int64_t w, int64_t h, int64_t p[4]) {
    const int64_t right = w - 1, bottom = h - 1;
    int           c[2] = {out_code(p[0], p[1], right, bottom),
                          out_code(p[2], p[3], right, bottom)};
    for (int stage = 0; stage < 2; stage++) {
        if ((c[0] & c[1]) != 0 || (c[0] | c[1]) == 0) {
            break;
        }
        const int u = stage;     // coordinate that moves: x, then y
        const int v = 1 - stage; // coordinate set to the edge
        for (int e = 0; e < 2; e++) {
            int64_t       *pe   = &p[2 * e];
            const int64_t *po   = &p[2 - 2 * e];
            const int      out  = c[e] & (stage ? 3 : 12);
            const int      high = c[e] & (stage ? 2 : 8); // right or below
            if (out) {
                clip_end(&pe[u], &pe[v], po[u], po[v],
                         high ? (stage ? right : bottom) : 0);
                c[e] = stage ? 0 : out_code(pe[0], 0, right, bottom);
            }
        }
    }
    return (c[0] | c[1]) == 0;
}

// Stamps the thickness-t brush (see draw.h) on (x, y). Brush row j spans
// |2j - e| <= t and, within it, |2i - e| <= isqrt(t^2 - (2j - e)^2). x and y
// lie within t / 2 <= 127 of the image, so every sum below fits int32_t
// (image sides are below INT32_MAX - 2^16). Only rows inside the image are
// visited.
static void stamp_brush(struct Mat *img, int32_t x, int32_t y, int32_t t,
                        const uchar *color, int cn) {
    const int32_t e   = (t + 1) & 1; // 1 for even t
    int32_t       jlo = -((t - e) / 2);
    int32_t       jhi = (t + e) / 2;
    if (y + jhi < 0 || y + jlo >= img->rows || x + jhi < 0 ||
        x + jlo >= img->cols) {
        return;
    }
    jlo = jlo < -y ? -y : jlo;
    jhi = jhi > img->rows - 1 - y ? img->rows - 1 - y : jhi;
    for (int32_t j = jlo; j <= jhi; j++) {
        const int32_t v = 2 * j - e;
        const int32_t m = (int32_t)isqrt_u64((uint64_t)(t * t - v * v));
        if (m < e) {
            continue; // no odd |2i - e| <= 0
        }
        hline_clipped(img, y + j, x - (m - e) / 2, x + (m + e) / 2, color, cn);
    }
}

void draw_line(struct Mat *img, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
               int thickness, const uchar color[4]) {
    const int cn = draw_channels(img, color);
    ASSERT(thickness >= 1 && thickness <= DRAW_MAX_THICKNESS);
    if (draw_is_empty(img)) {
        return;
    }
    // Clip to the image grown by the brush reach m on every side (m = 0 for
    // thickness 1); the walk then runs in the grown frame, coordinates + m.
    const int32_t t    = thickness;
    const int32_t m    = t / 2;
    int64_t       p[4] = {(int64_t)x0 + m, (int64_t)y0 + m, (int64_t)x1 + m,
                          (int64_t)y1 + m};
    if (!clip_line((int64_t)img->cols + 2 * m, (int64_t)img->rows + 2 * m, p)) {
        return;
    }
    // Both clipped ends lie in the grown image, so they fit int32_t.
    int32_t ax = (int32_t)p[0], ay = (int32_t)p[1];
    int32_t bx = (int32_t)p[2], by = (int32_t)p[3];
    if (bx < ax) { // walk from the left end, as cv2.LineIterator
        const int32_t tx = ax, ty = ay;
        ax = bx;
        ay = by;
        bx = tx;
        by = ty;
    }
    int32_t       dx   = bx - ax;
    int32_t       dy   = by - ay;
    const int32_t sy   = dy < 0 ? -1 : 1;
    const int     vert = (dy < 0 ? -dy : dy) > dx;
    dy                 = dy < 0 ? -dy : dy;
    if (vert) {
        const int32_t tmp = dx;
        dx                = dy;
        dy                = tmp;
    }
    // e is half the classic error dx - 2 * dy (whose parity is that of dx,
    // so err < 0 exactly when e < 0) and stays in [-dy, dx - dy].
    int32_t e = dx / 2 - dy;
    int32_t x = ax, y = ay;
    for (int32_t i = dx;; i--) {
        if (t == 1) {
            put_px(img, (int)x, (int)y, color, cn);
        } else {
            stamp_brush(img, x - m, y - m, t, color, cn);
        }
        if (i == 0) {
            break;
        }
        if (e < 0) {
            e += dx - dy;
            x += 1;
            y += sy;
        } else {
            e -= dy;
            if (vert) {
                y += sy;
            } else {
                x += 1;
            }
        }
    }
}

// ---- Circles ----------------------------------------------------------------

// The eight points (+-g, +-k) and (+-k, +-g) around (cx, cy).
static void plot8(struct Mat *img, int64_t cx, int64_t cy, int64_t g, int64_t k,
                  const uchar *color, int cn) {
    for (int q = 0; q < 8; q++) {
        const int64_t u = (q & 4) ? k : g;
        const int64_t v = (q & 4) ? g : k;
        put_clipped(img, (q & 1) ? cx + u : cx - u, (q & 2) ? cy + v : cy - v,
                    color, cn);
    }
}

// Thickness-1 outline. The (+-g, +-k) points of step k can be visible only
// when cy - k or cy + k is a row of the image, and the (+-k, +-g) points only
// when cx - k or cx + k is a column, so only those k are visited: at most
// 2 * (rows + cols) steps, whatever the radius.
static void circle_outline(struct Mat *img, int32_t cx, int32_t cy, int32_t r,
                           const uchar *color, int cn) {
    const uint64_t r2   = (uint64_t)r * (uint64_t)r;
    const int64_t  kmax = (int64_t)isqrt_u64(r2 / 2); // 2k^2 <= r^2, < 2^31
    for (int s = 0; s < 4; s++) {
        // s 0, 1: rows cy - k, cy + k; s 2, 3: columns cx - k, cx + k. The
        // k that put that row or column inside 0..n-1 form one interval.
        const int64_t c  = s < 2 ? cy : cx;
        const int64_t n  = s < 2 ? img->rows : img->cols;
        int64_t       lo = (s & 1) ? -c : c - (n - 1);
        int64_t       hi = (s & 1) ? n - 1 - c : c;
        lo               = lo < 0 ? 0 : lo;
        hi               = hi > kmax ? kmax : hi;
        if (lo > hi) {
            continue;
        }
        // From here on k and g fit uint32_t; only the squares need 64 bits.
        const uint32_t a = (uint32_t)lo, b = (uint32_t)hi;
        uint32_t       g = (uint32_t)isqrt_u64(r2 - (uint64_t)a * a);
        for (uint32_t k = a; k <= b; k++) {
            while ((uint64_t)g * g + (uint64_t)k * k > r2) {
                g--; // g falls by at most 1 while k <= g
            }
            plot8(img, cx, cy, g, k, color, cn);
        }
    }
}

// Pixels with dx^2 + dy^2 <= outer2, minus those with dx^2 + dy^2 <= hole2
// when has_hole, one image row at a time. outer2 < 2^64.
static void circle_rows(struct Mat *img, int64_t cx, int64_t cy,
                        uint64_t outer2, int has_hole, uint64_t hole2,
                        const uchar *color, int cn) {
    const int64_t reach = (int64_t)isqrt_u64(outer2); // largest |dy|
    int64_t       lo = cy - reach, hi = cy + reach;
    lo             = lo < 0 ? 0 : lo;
    hi             = hi > (int64_t)img->rows - 1 ? (int64_t)img->rows - 1 : hi;
    const int cols = img->cols;
    for (int32_t y = (int32_t)lo; y <= (int32_t)hi; y++) {
        const uint64_t dy = abs_u64((int64_t)y - cy);
        const uint64_t d2 = dy * dy; // <= outer2
        const int64_t  go = (int64_t)isqrt_u64(outer2 - d2);
        // Hole columns cx - gi .. cx + gi; gi = -1 (empty) without a hole.
        const int64_t gi =
            has_hole && d2 <= hole2 ? (int64_t)isqrt_u64(hole2 - d2) : -1;
        const int32_t a = clamp_col(cx - go, cols);
        const int32_t b = clamp_col(cx - gi - 1, cols);
        const int32_t c = clamp_col(cx + gi + 1, cols);
        const int32_t d = clamp_col(cx + go, cols);
        if (gi < 0) {
            hline_clipped(img, (int32_t)y, a, d, color, cn);
        } else {
            hline_clipped(img, (int32_t)y, a, b, color, cn);
            hline_clipped(img, (int32_t)y, c, d, color, cn);
        }
    }
}

void draw_circle(struct Mat *img, int32_t cx, int32_t cy, int32_t radius,
                 int thickness, const uchar color[4]) {
    const int cn = draw_channels(img, color);
    ASSERT(radius >= 0);
    ASSERT(thickness != 0 && thickness <= DRAW_MAX_THICKNESS);
    if (draw_is_empty(img)) {
        return;
    }
    if (thickness < 0) {
        const uint64_t r = (uint64_t)radius;
        circle_rows(img, cx, cy, r * r, 0, 0, color, cn);
    } else if (thickness == 1) {
        circle_outline(img, cx, cy, radius, color, cn);
    } else {
        // Ring between the half-integer radii ri + 1/2 and ro + 1/2:
        // ri^2 + ri < d^2 <= ro^2 + ro. ro < 2^32, so ro^2 + ro < 2^64.
        const int64_t  ro = (int64_t)radius + thickness / 2;
        const int64_t  ri = ro - thickness;
        const uint64_t uo = (uint64_t)ro;
        const uint64_t ui = ri >= 0 ? (uint64_t)ri : 0;
        circle_rows(img, cx, cy, uo * uo + uo, ri >= 0, ui * ui + ui, color,
                    cn);
    }
}

// ---- Text -------------------------------------------------------------------

// 5x8 glyphs for 0x20..0x7E, then the replacement box. One byte per column,
// left to right; bit k is row k (row 0 at the top). Drawn for this project as
// ASCII art and converted by a script, not taken from a font file. A plain
// const table like the others in core/: on AVR it occupies 480 bytes of RAM
// unless a port moves it to PROGMEM; --gc-sections drops it when draw_text is
// not used.
static const uint8_t draw_font[96][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, // 0x20 space
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // 0x21 !
    {0x00, 0x03, 0x00, 0x03, 0x00}, // 0x22 "
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, // 0x23 #
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // 0x24 $
    {0x23, 0x13, 0x08, 0x64, 0x62}, // 0x25 %
    {0x36, 0x49, 0x55, 0x22, 0x50}, // 0x26 &
    {0x00, 0x00, 0x03, 0x00, 0x00}, // 0x27 apostrophe
    {0x00, 0x1C, 0x22, 0x41, 0x00}, // 0x28 (
    {0x00, 0x41, 0x22, 0x1C, 0x00}, // 0x29 )
    {0x14, 0x08, 0x3E, 0x08, 0x14}, // 0x2A *
    {0x08, 0x08, 0x3E, 0x08, 0x08}, // 0x2B +
    {0x00, 0xA0, 0x60, 0x00, 0x00}, // 0x2C ,
    {0x08, 0x08, 0x08, 0x08, 0x08}, // 0x2D -
    {0x00, 0x60, 0x60, 0x00, 0x00}, // 0x2E .
    {0x20, 0x10, 0x08, 0x04, 0x02}, // 0x2F /
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0x30 0
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 0x31 1
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 0x32 2
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 0x33 3
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 0x34 4
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 0x35 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 0x36 6
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 0x37 7
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 0x38 8
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 0x39 9
    {0x00, 0x36, 0x36, 0x00, 0x00}, // 0x3A :
    {0x00, 0xB6, 0x76, 0x00, 0x00}, // 0x3B ;
    {0x08, 0x14, 0x22, 0x41, 0x00}, // 0x3C <
    {0x14, 0x14, 0x14, 0x14, 0x14}, // 0x3D =
    {0x00, 0x41, 0x22, 0x14, 0x08}, // 0x3E >
    {0x02, 0x01, 0x51, 0x09, 0x06}, // 0x3F ?
    {0x3E, 0x41, 0x5D, 0x55, 0x1E}, // 0x40 @
    {0x7E, 0x09, 0x09, 0x09, 0x7E}, // 0x41 A
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // 0x42 B
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // 0x43 C
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // 0x44 D
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // 0x45 E
    {0x7F, 0x09, 0x09, 0x09, 0x01}, // 0x46 F
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, // 0x47 G
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // 0x48 H
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // 0x49 I
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // 0x4A J
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // 0x4B K
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // 0x4C L
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // 0x4D M
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // 0x4E N
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // 0x4F O
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // 0x50 P
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // 0x51 Q
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // 0x52 R
    {0x46, 0x49, 0x49, 0x49, 0x31}, // 0x53 S
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // 0x54 T
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // 0x55 U
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // 0x56 V
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, // 0x57 W
    {0x63, 0x14, 0x08, 0x14, 0x63}, // 0x58 X
    {0x03, 0x04, 0x78, 0x04, 0x03}, // 0x59 Y
    {0x61, 0x51, 0x49, 0x45, 0x43}, // 0x5A Z
    {0x00, 0x7F, 0x41, 0x41, 0x00}, // 0x5B [
    {0x02, 0x04, 0x08, 0x10, 0x20}, // 0x5C backslash
    {0x00, 0x41, 0x41, 0x7F, 0x00}, // 0x5D ]
    {0x04, 0x02, 0x01, 0x02, 0x04}, // 0x5E ^
    {0x80, 0x80, 0x80, 0x80, 0x80}, // 0x5F _
    {0x00, 0x01, 0x02, 0x04, 0x00}, // 0x60 `
    {0x20, 0x54, 0x54, 0x54, 0x78}, // 0x61 a
    {0x7F, 0x48, 0x44, 0x44, 0x38}, // 0x62 b
    {0x38, 0x44, 0x44, 0x44, 0x20}, // 0x63 c
    {0x38, 0x44, 0x44, 0x48, 0x7F}, // 0x64 d
    {0x38, 0x54, 0x54, 0x54, 0x18}, // 0x65 e
    {0x08, 0x7E, 0x09, 0x01, 0x02}, // 0x66 f
    {0x18, 0xA4, 0xA4, 0xA4, 0x7C}, // 0x67 g
    {0x7F, 0x08, 0x04, 0x04, 0x78}, // 0x68 h
    {0x00, 0x44, 0x7D, 0x40, 0x00}, // 0x69 i
    {0x40, 0x80, 0x84, 0x7D, 0x00}, // 0x6A j
    {0x7F, 0x10, 0x28, 0x44, 0x00}, // 0x6B k
    {0x00, 0x41, 0x7F, 0x40, 0x00}, // 0x6C l
    {0x7C, 0x04, 0x78, 0x04, 0x78}, // 0x6D m
    {0x7C, 0x08, 0x04, 0x04, 0x78}, // 0x6E n
    {0x38, 0x44, 0x44, 0x44, 0x38}, // 0x6F o
    {0xFC, 0x24, 0x24, 0x24, 0x18}, // 0x70 p
    {0x18, 0x24, 0x24, 0x24, 0xFC}, // 0x71 q
    {0x7C, 0x08, 0x04, 0x04, 0x08}, // 0x72 r
    {0x48, 0x54, 0x54, 0x54, 0x24}, // 0x73 s
    {0x04, 0x3F, 0x44, 0x40, 0x20}, // 0x74 t
    {0x3C, 0x40, 0x40, 0x20, 0x7C}, // 0x75 u
    {0x1C, 0x20, 0x40, 0x20, 0x1C}, // 0x76 v
    {0x3C, 0x40, 0x30, 0x40, 0x3C}, // 0x77 w
    {0x44, 0x28, 0x10, 0x28, 0x44}, // 0x78 x
    {0x1C, 0xA0, 0xA0, 0xA0, 0x7C}, // 0x79 y
    {0x44, 0x64, 0x54, 0x4C, 0x44}, // 0x7A z
    {0x00, 0x08, 0x36, 0x41, 0x41}, // 0x7B {
    {0x00, 0x00, 0x7F, 0x00, 0x00}, // 0x7C |
    {0x41, 0x41, 0x36, 0x08, 0x00}, // 0x7D }
    {0x08, 0x04, 0x08, 0x10, 0x08}, // 0x7E ~
    {0x7F, 0x41, 0x41, 0x41, 0x7F}, // replacement box
};

#define DRAW_GLYPH_BOX 95

static int is_continuation(unsigned char c) { return c >= 0x80 && c <= 0xBF; }

static int glyph_index(unsigned char c) {
    return c >= 0x20 && c <= 0x7E ? c - 0x20 : DRAW_GLYPH_BOX;
}

// Draws one glyph with its top-left corner at (gx, gy); each font pixel is a
// scale x scale block, clipped to the image. The caller keeps the glyph cell
// overlapping the image, so gx and gy are within 8192 of it and every sum
// fits int32_t (image sides are below INT32_MAX - 2^16).
static void draw_glyph(struct Mat *img, const uint8_t glyph[5], int32_t gx,
                       int32_t gy, int32_t scale, const uchar *color, int cn) {
    for (int col = 0; col < 5; col++) {
        const int32_t bx = gx + col * scale;
        for (int row = 0; row < 8; row++) {
            if (((glyph[col] >> row) & 1u) == 0) {
                continue;
            }
            const int32_t by = gy + row * scale;
            for (int32_t k = 0; k < scale; k++) {
                hline_clipped(img, by + k, bx, bx + scale - 1, color, cn);
            }
        }
    }
}

void draw_text(struct Mat *img, const char *text, int32_t x, int32_t y,
               int scale, const uchar color[4]) {
    const int cn = draw_channels(img, color);
    ASSERT(text != NULL);
    ASSERT(scale >= 1 && scale <= DRAW_TEXT_MAX_SCALE);
    const int32_t cell = (int32_t)DRAW_FONT_W * scale;
    if (draw_is_empty(img) || y >= img->rows ||
        y <= -(int32_t)DRAW_FONT_H * scale) {
        return;
    }
    int32_t pen = x; // left column of the next cell
    for (const unsigned char *s = (const unsigned char *)text; *s != 0; s++) {
        if (is_continuation(*s)) {
            continue;
        }
        if (pen >= img->cols) {
            break; // this and every later cell is right of the image
        }
        if (pen > -cell) { // the cell overlaps the image columns
            draw_glyph(img, draw_font[glyph_index(*s)], pen, y, scale, color,
                       cn);
        }
        if (pen > img->cols - cell) {
            break; // the next cell would start right of the image
        }
        pen += cell;
    }
}

int32_t draw_text_width(const char *text, int scale) {
    ASSERT(text != NULL);
    ASSERT(scale >= 1 && scale <= DRAW_TEXT_MAX_SCALE);
    const int32_t cell  = (int32_t)DRAW_FONT_W * scale;
    int32_t       cells = 0;
    for (const unsigned char *s = (const unsigned char *)text; *s != 0; s++) {
        if (!is_continuation(*s)) {
            ASSERT(cells < INT32_MAX / cell);
            cells++;
        }
    }
    return cells * cell;
}
