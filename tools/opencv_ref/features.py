"""OpenCV references for the feature Kodak ops (tests/kodak_ops_features.c).

Each op is a CV_8UC1 map of FAST-9 corners on our gray image: with non-max
suppression the pixel holds the response, without it 255; other pixels are 0.
cv2 and fast_detect must give the same points and responses, so the
tolerance is 0.
"""

import cv2
import numpy as np

THRESHOLD = 20  # KODAK_FAST_THRESHOLD in tests/kodak_ops_features.c


def _fast_map(ctx, nonmax):
    gray = ctx.get("gray")
    det = cv2.FastFeatureDetector_create(THRESHOLD, nonmax,
                                         cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    out = np.zeros_like(gray)
    for k in det.detect(gray):
        x, y = int(k.pt[0]), int(k.pt[1])
        out[y, x] = int(k.response) if nonmax else 255
    return out


OPS = {
    "fast": (lambda c: _fast_map(c, True), 0, "FastFeatureDetector 9_16, t=20, NMS"),
    "fast_raw": (lambda c: _fast_map(c, False), 0, "FastFeatureDetector 9_16, t=20, no NMS"),
}
