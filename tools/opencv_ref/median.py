"""OpenCV references for the median Kodak ops (tests/kodak_ops_median.c).

All tolerances are 0. cv2.medianBlur uses BORDER_REPLICATE and returns the
element of rank ksize^2 / 2 of each window, as core/filter.c does.
"""

import cv2


def _median(k, src="src"):
    return lambda c: cv2.medianBlur(c.get(src), k)


OPS = {
    "median3": (_median(3), 0, "cv2.medianBlur 3 on RGB"),
    "median5": (_median(5), 0, "cv2.medianBlur 5 on RGB"),
    "gray_median7": (_median(7, "median_gray_src"), 0, "cv2.medianBlur 7 on our gray"),
    "gray_median15": (_median(15, "median_gray_src"), 0, "cv2.medianBlur 15 on our gray"),
}
