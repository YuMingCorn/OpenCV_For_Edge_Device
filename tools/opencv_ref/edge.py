"""OpenCV references for the edge-device motion Kodak ops (tests/kodak_ops_edge.c).

Frame 0 is the image without its last SHIFT rows and columns. Frame 1 is a copy
of frame 0 with one patch replaced by the content SHIFT pixels down and right:
the middle half (motion_*), a CORNER x CORNER patch at the top left (corner_*),
or no patch (still_stats). crop, absdiff, threshold and the nonzero statistics
are compared exactly (tolerance 0). Every gray op reads our own dumped
edge_gray, so a gray difference never leaks in.

The *_stats outputs are not images (values_edge.txt). motion_box checks
motion_stats and still_stats, and corner_box checks corner_stats, against
numpy; a missing or different line raises an error.
"""

import os

import cv2
import numpy as np

SHIFT = 6
MOTION_T = 40
CROSS_HALF = 6
CORNER = 12


def _center(img):
    h, w = img.shape[0] - SHIFT, img.shape[1] - SHIFT
    return (w // 4, h // 4, w // 2, h // 2)


def _frames(img, patch):
    h, w = img.shape[0] - SHIFT, img.shape[1] - SHIFT
    f0 = img[:h, :w]
    f1 = f0.copy()
    x, y, pw, ph = patch
    if pw and ph:
        f1[y:y + ph, x:x + pw] = img[y + SHIFT:y + SHIFT + ph, x + SHIFT:x + SHIFT + pw]
    return f0, f1


def _crop_center(c):
    src = c.get("src")
    h, w = src.shape[:2]
    return src[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2]


def _absdiff_shift(c):
    src = c.get("src")
    h, w = src.shape[0] - SHIFT, src.shape[1] - SHIFT
    f0, f1 = _frames(src, (0, 0, w, h))
    return cv2.absdiff(f0, f1)


def _mask(c, patch):
    f0, f1 = _frames(c.get("edge_gray"), patch)
    return cv2.threshold(cv2.absdiff(f0, f1), MOTION_T, 255, cv2.THRESH_BINARY)[1]


def _stats(mask):
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if n == 0:
        return [0, 0, 0, -1, -1, -1, -1]
    cx = (int(xs.sum()) + n // 2) // n
    cy = (int(ys.sum()) + n // 2) // n
    return [n, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), cx, cy]


def _check_stats(c, op, expected):
    path = os.path.join(c.dump_dir, "values_edge.txt")
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts[0] == c.image and parts[1] == op:
                ours = [int(v) for v in parts[5:]]
                if ours != expected:
                    raise ValueError(f"{c.image}: {op} {ours} != numpy {expected}")
                return
    raise ValueError(f"{c.image}: no {op} line in {path}")


def _draw(c, patch, stats):
    n, x0, y0, x1, y1, cx, cy = stats
    out = _frames(c.get("src"), patch)[1].copy()
    h, w = out.shape[:2]
    if n > 0:
        out[y0, x0:x1 + 1] = (255, 0, 0)
        out[y1, x0:x1 + 1] = (255, 0, 0)
        out[y0:y1 + 1, x0] = (255, 0, 0)
        out[y0:y1 + 1, x1] = (255, 0, 0)
        out[cy, max(cx - CROSS_HALF, 0):min(cx + CROSS_HALF, w - 1) + 1] = (0, 255, 0)
        out[max(cy - CROSS_HALF, 0):min(cy + CROSS_HALF, h - 1) + 1, cx] = (0, 255, 0)
    return out


def _motion_box(c):
    patch = _center(c.get("src"))
    stats = _stats(_mask(c, patch))
    _check_stats(c, "motion_stats", stats)
    _check_stats(c, "still_stats", _stats(_mask(c, (0, 0, 0, 0))))
    return _draw(c, patch, stats)


def _corner_box(c):
    patch = (0, 0, CORNER, CORNER)
    stats = _stats(_mask(c, patch))
    _check_stats(c, "corner_stats", stats)
    return _draw(c, patch, stats)


OPS = {
    "edge_gray": (lambda c: cv2.cvtColor(c.get("src"), cv2.COLOR_RGB2GRAY), 1,
                  "COLOR_RGB2GRAY"),
    "crop_center": (_crop_center, 0, "numpy slice of the middle half"),
    "absdiff_shift": (_absdiff_shift, 0, "cv2.absdiff of two frames 6 px apart"),
    "motion_mask": (lambda c: _mask(c, _center(c.get("src"))), 0,
                    "absdiff on our gray, THRESH_BINARY 40, middle patch moved"),
    "motion_box": (_motion_box, 0,
                   "numpy bbox + centroid; also checks motion_stats, still_stats"),
    "corner_box": (_corner_box, 0,
                   "12x12 corner patch, cross clipped; also checks corner_stats"),
}
