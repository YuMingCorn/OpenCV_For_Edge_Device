"""OpenCV references for the drawing Kodak ops (tests/kodak_ops_draw.c).

Each op draws on a copy of our dumped src with the same shapes, colours and
order as the C file. draw_lines and draw_circles use cv2.line and cv2.circle
(LINE_8, thickness 1 and filled), which our code matches exactly (tolerance 0;
every coordinate is within +-2^26, where cv2's double clipping is exact).

draw_thick has no OpenCV equivalent: cv2 draws a thickness-t line or circle
about 2 * ceil(t / 2) + 1 pixels wide with a polygon fill. The reference is a
model of the rules in core/draw.h built from cv2 calls: a thick line is the
cv2 thickness-1 line walked in the image grown by t / 2 and dilated with the
round brush; a thick circle is a numpy ring between the radii ri + 1/2 and
ro + 1/2.

draw_text has no reference (cv2.putText uses vector fonts); it is absent from
OPS and checked by the Kodak hash only.
"""

import cv2
import numpy as np

MARGIN = 37
STEP = 53
BIG = 1 << 24
HUGE_R = 1 << 20


def _color(i):
    return ((i * 67 + 20) & 255, (i * 29 + 80) & 255, (255 - i * 13) & 255)


def _lines(c):
    out = c.get("src").copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    i = 0

    def line(x0, y0, x1, y1):
        nonlocal i
        cv2.line(out, (x0, y0), (x1, y1), _color(i), 1, cv2.LINE_8)
        i += 1

    for x in range(-MARGIN, w + MARGIN + 1, STEP):
        line(cx, cy, x, -MARGIN)
        line(cx, cy, x, h - 1 + MARGIN)
    for y in range(-MARGIN, h + MARGIN + 1, STEP):
        line(cx, cy, -MARGIN, y)
        line(cx, cy, w - 1 + MARGIN, y)
    line(-50, -10, -5, h + 100)
    line(2, 2, 9, 5)
    line(5, 9, 5, 9)
    line(w - 1, h - 1, w - 30, h - 4)
    line(-BIG, h // 3, BIG, h // 3 + 5)
    line(w // 3, -BIG, w // 3 + 9, BIG)
    line(-BIG, -BIG + 3, BIG, BIG)
    line(BIG, -BIG, -BIG + 11, BIG)
    return out


def _circles(c):
    out = c.get("src").copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    rmax = min(w, h) // 2 + 40
    i = 0

    def circle(x, y, r, t):
        nonlocal i
        cv2.circle(out, (x, y), r, _color(i), t, cv2.LINE_8)
        i += 1

    circle(0, 0, 45, -1)
    circle(w - 1, h - 1, 60, -1)
    circle(w + 10, cy, 30, -1)
    circle(-100, -100, 20, -1)
    circle(cx, -HUGE_R, HUGE_R + 30, -1)
    for r in range(0, rmax + 1, 9):
        circle(cx, cy, r, 1)
    for r in range(10, 71, 15):
        circle(w - 20, 15, r, 1)
    circle(cx, cy + HUGE_R, HUGE_R - h // 4, 1)
    circle(7, 7, 3, -1)
    return out


def _brush_kernel(t):
    """Dilation kernel of the thickness-t brush, anchor (hi, hi).

    The brush holds the offsets (i, j) with (2i - e)^2 + (2j - e)^2 <= t^2
    (e = 1 for even t). cv2.dilate reads src(q + k - anchor), so the kernel
    is the brush mirrored: k = hi - offset.
    """
    e = (t + 1) & 1
    lo, hi = -((t - e) // 2), (t + e) // 2
    k = np.zeros((hi - lo + 1, hi - lo + 1), np.uint8)
    for j in range(lo, hi + 1):
        for i in range(lo, hi + 1):
            if (2 * i - e) ** 2 + (2 * j - e) ** 2 <= t * t:
                k[hi - j, hi - i] = 1
    return k, hi


def _thick_line(out, p0, p1, t, color):
    h, w = out.shape[:2]
    g = t // 2
    path = np.zeros((h + 2 * g, w + 2 * g), np.uint8)
    cv2.line(path, (p0[0] + g, p0[1] + g), (p1[0] + g, p1[1] + g), 255, 1,
             cv2.LINE_8)
    kernel, hi = _brush_kernel(t)
    mask = cv2.dilate(path, kernel, anchor=(hi, hi),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Brush centres in the grown border still reach the image.
    out[mask[g:g + h, g:g + w] > 0] = color


def _ring(out, cx, cy, r, t, color):
    h, w = out.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ro = r + t // 2
    ri = ro - t
    mask = d2 <= ro * ro + ro  # radius ro + 1/2
    if ri >= 0:
        mask &= d2 > ri * ri + ri  # outside radius ri + 1/2
    out[mask] = color


def _thick(c):
    out = c.get("src").copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    i = 0
    for k in range(8):
        t = k + 2
        _thick_line(out, (w * k // 8, -20), (w - 1 - w * k // 8 + 15, h + 20),
                    t, _color(i))
        i += 1
        _thick_line(out, (-30, h * k // 8 + 10), (w + 30, h * k // 8 + 40),
                    t, _color(i))
        i += 1
    for k in range(5):
        _ring(out, cx, cy, 30 + 25 * k, k + 2, _color(i))
        i += 1
    _ring(out, w - 1, 0, 80, 7, _color(i))
    i += 1
    _thick_line(out, (20, h - 30), (w - 40, 30), 25, _color(i))
    i += 1
    _thick_line(out, (40, 40), (40, 40), 15, _color(i))
    i += 1
    _thick_line(out, (-BIG, cy), (BIG, cy + 3), 4, _color(i))
    return out


OPS = {
    "draw_lines": (_lines, 0, "cv2.line LINE_8, thickness 1"),
    "draw_circles": (_circles, 0, "cv2.circle LINE_8, thickness 1 and -1"),
    "draw_thick": (_thick, 0, "own thick rules: cv2.line + dilate, ring"),
}
