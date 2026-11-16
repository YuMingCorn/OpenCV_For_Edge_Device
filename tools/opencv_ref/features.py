"""OpenCV references for the feature Kodak ops (tests/kodak_ops_features.c).

fast, fast_raw: CV_8UC1 maps of FAST-9 corners on our gray image. With
non-max suppression the pixel holds the response, without it 255; other
pixels are 0. cv2 and fast_detect must give the same points and responses, so
the tolerance is 0.

orb_rot30: our ORB is not bit-compatible with cv2.ORB (own test pairs, 32
integer angle steps), so it is compared on matching quality instead. Both run
on our gray image and on our 30-degree rotation of it (the rot30 dump), with
500 keypoints, one scale, FAST threshold 20 and cross-checked Hamming
matching. A match is correct when it lands within 3 pixels of the true
rotated position. The values are not images, so they are checked inside the
reference of "rot30", which runs once per image: our correct share must not
fall more than 10 points below cv2's on any image. Run this file directly for
the per-image table and the total, which must not fall more than 3 points
below cv2's:

    python3 tools/opencv_ref/features.py --dump <dump dir>
"""

import argparse
import os
import sys

import cv2
import numpy as np

THRESHOLD = 20  # KODAK_FAST_THRESHOLD in tests/kodak_ops_features.c
ORB_KPS = 500   # KODAK_ORB_KPS
COS30 = 56756 / 65536.0  # ROT_COS_Q16
SIN30 = 32768 / 65536.0  # ROT_SIN_Q16
PER_IMAGE_MARGIN = 0.10
TOTAL_MARGIN = 0.03


def _fast_map(ctx, nonmax):
    gray = ctx.get("gray")
    det = cv2.FastFeatureDetector_create(THRESHOLD, nonmax,
                                         cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    out = np.zeros_like(gray)
    for k in det.detect(gray):
        x, y = int(k.pt[0]), int(k.pt[1])
        out[y, x] = int(k.response) if nonmax else 255
    return out


def read_values(dump_dir):
    """Returns {(image, op): list of ints} from values_features.txt."""
    values = {}
    with open(os.path.join(dump_dir, "values_features.txt")) as f:
        for line in f:
            parts = line.split()
            values[(parts[0], parts[1])] = [int(v) for v in parts[5:]]
    return values


def rotate_points(pts, shape):
    """The forward map of op_rot30: R(30 degrees) about the image centre."""
    rows, cols = shape
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([dx * COS30 - dy * SIN30 + cx, dx * SIN30 + dy * COS30 + cy], 1)


def cv_orb_rot30(gray, rot):
    """(matches, correct) of cv2.ORB between gray and its rotation."""
    orb = cv2.ORB_create(ORB_KPS, nlevels=1, fastThreshold=THRESHOLD)
    ka, da = orb.detectAndCompute(gray, None)
    kb, db = orb.detectAndCompute(rot, None)
    if da is None or db is None:
        return 0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if not m:
        return 0, 0
    pa = np.array([ka[x.queryIdx].pt for x in m])
    pb = np.array([kb[x.trainIdx].pt for x in m])
    err = np.linalg.norm(rotate_points(pa, gray.shape) - pb, axis=1)
    return len(m), int((err < 3).sum())


def check_rot30(ctx):
    """Returns (ours_matches, ours_correct, cv_matches, cv_correct)."""
    ours = read_values(ctx.dump_dir)[(ctx.image, "orb_rot30")]
    cv_m, cv_c = cv_orb_rot30(ctx.get("gray"), ctx.get("rot30"))
    return ours[0], ours[1], cv_m, cv_c


def share(correct, matches):
    return correct / matches if matches else 0.0


def _rot30(ctx):
    om, oc, cm, cc = check_rot30(ctx)
    if share(oc, om) < share(cc, cm) - PER_IMAGE_MARGIN:
        sys.exit(f"{ctx.image}: ORB correct share {oc}/{om} is more than "
                 f"{PER_IMAGE_MARGIN:.0%} below cv2.ORB {cc}/{cm}")
    return ctx.get("rot30")  # our own rotation: the check above is the test


OPS = {
    "fast": (lambda c: _fast_map(c, True), 0, "FastFeatureDetector 9_16, t=20, NMS"),
    "fast_raw": (lambda c: _fast_map(c, False), 0, "FastFeatureDetector 9_16, t=20, no NMS"),
    "rot30": (_rot30, 0, "our rotation; checks orb_rot30 against cv2.ORB matching"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from compare_opencv import Context, find_images

    print(f"{'image':<9}{'ours':>12}{'%':>7}{'cv2.ORB':>12}{'%':>7}")
    tot = [0, 0, 0, 0]
    for image in find_images(args.dump):
        r = check_rot30(Context(args.dump, image))
        tot = [a + b for a, b in zip(tot, r)]
        print(f"{image:<9}{r[1]:>5}/{r[0]:<6}{100 * share(r[1], r[0]):>6.1f}"
              f"{r[3]:>5}/{r[2]:<6}{100 * share(r[3], r[2]):>6.1f}")
    ours, ref = share(tot[1], tot[0]), share(tot[3], tot[2])
    print(f"{'total':<9}{tot[1]:>5}/{tot[0]:<6}{100 * ours:>6.1f}"
          f"{tot[3]:>5}/{tot[2]:<6}{100 * ref:>6.1f}")
    ok = ours >= ref - TOTAL_MARGIN
    print("ok" if ok else f"FAIL: more than {TOTAL_MARGIN:.0%} below cv2.ORB")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
