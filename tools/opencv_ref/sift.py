"""OpenCV references for the SIFT Kodak ops (tests/kodak_ops_sift.c).

half: our gray image halved with INTER_AREA; the input of every SIFT op.

Every cv2 call here runs with cv2.setUseOptimized(False): the baseline SSE
code without FMA, whose float operation order core/sift.c follows. Our SIFT
then matches cv2.SIFT bit for bit (cv2 4.10 on x86-64 glibc, the only one
measured: core/sift.c ports glibc's powf, sinf and cosf). IMPROC_CV_OPTIMIZED=1 compares against cv2's
default dispatch instead (AVX2 and FMA), with the loose checks below.

sift_desc: our descriptors (one row per keypoint, in the sift_kps order)
against cv2.SIFT.compute at our own keypoints, so the two implementations
are compared on the same points. Its reference also checks sift_kps: our
keypoints (x, y, size, angle, response) must be exactly cv2.SIFT.detect's.
With IMPROC_CV_OPTIMIZED=1, the descriptors may differ by 1 and the keypoint
check is: at least 95 % of the cv2.SIFT keypoints must have one of ours
within 1 pixel (size ratio 0.8..1.25, angle within 10 degrees), and the
other way round.

rotscale: our 30-degree rotation and 0.75 scale of the half image. Its
reference checks sift_rotscale: our correct share of ratio-test matches must
not fall more than 5 points below cv2.SIFT's on any image. Run this file
directly for the per-image tables:
    python3 tools/opencv_ref/sift.py --dump <dump dir>
"""

import argparse
import collections
import contextlib
import os
import sys

import cv2
import numpy as np

FWD_COS = 42567 / 65536.0  # FWD_COS_Q16 in tests/kodak_ops_sift.c
FWD_SIN = 24576 / 65536.0  # FWD_SIN_Q16
RATIO = 0.8
AGREE_MIN = 0.95
PER_IMAGE_MARGIN = 0.05
OPTIMIZED = os.environ.get("IMPROC_CV_OPTIMIZED", "0") == "1"


@contextlib.contextmanager
def cv_mode():
    """Runs cv2 in the compared mode, then restores it for the other modules."""
    old = cv2.useOptimized()
    cv2.setUseOptimized(OPTIMIZED)
    try:
        yield
    finally:
        cv2.setUseOptimized(old)


def read_values(dump_dir):
    """Returns {(image, op): (rows, cols, list of floats)} from values_sift.txt."""
    values = {}
    with open(os.path.join(dump_dir, "values_sift.txt")) as f:
        for line in f:
            p = line.split()
            values[(p[0], p[1])] = (int(p[2]), int(p[3]), [float(v) for v in p[5:]])
    return values


def our_kps(ctx):
    """N x 7 array: x, y, size, angle, response, octave, layer."""
    rows, cols, v = read_values(ctx.dump_dir)[(ctx.image, "sift_kps")]
    return np.array(v, dtype=np.float64).reshape(rows, cols)


def to_cv(k):
    kp = cv2.KeyPoint(float(k[0]), float(k[1]), float(k[2]), float(k[3]),
                      float(k[4]))
    kp.octave = (int(k[5]) & 255) | (int(k[6]) << 8)
    return kp


def _angle_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def agreement(a, b):
    """Share of keypoints in a with a keypoint of b within 1 pixel, size
    ratio 0.8..1.25 and angle within 10 degrees. Rows are x, y, size, angle."""
    if len(a) == 0:
        return 1.0
    hit = 0
    for k in a:
        near = np.hypot(b[:, 0] - k[0], b[:, 1] - k[1]) <= 1.0
        ratio = k[2] / b[:, 2]
        ok = near & (ratio > 0.8) & (ratio < 1.25) & (_angle_diff(k[3], b[:, 3]) < 10)
        hit += bool(ok.any())
    return hit / len(a)


def cv_detect(half):
    with cv_mode():
        ref = cv2.SIFT_create().detect(half, None)
    return np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response] for k in ref],
                    dtype=np.float32).reshape(-1, 5)


def check_kps(ctx):
    """(recall, precision, ours, cv2, exact): exact is the number of our
    keypoints that equal a cv2 keypoint in all five float32 values."""
    ours = our_kps(ctx)
    ref = cv_detect(ctx.get("half"))
    left = collections.Counter(r.tobytes() for r in ref)
    exact = 0
    for k in ours[:, :5].astype(np.float32):
        if left[k.tobytes()] > 0:
            left[k.tobytes()] -= 1
            exact += 1
    ref4 = ref[:, :4].astype(np.float64)
    return (agreement(ref4, ours[:, :4]), agreement(ours[:, :4], ref4), len(ours),
            len(ref), exact)


def _sift_desc(ctx):
    rec, prec, no, nc, exact = check_kps(ctx)
    if not OPTIMIZED and not no == nc == exact:
        sys.exit(f"{ctx.image}: {exact} of our {no} SIFT keypoints equal one of "
                 f"cv2's {nc}; all must")
    if rec < AGREE_MIN or prec < AGREE_MIN:
        sys.exit(f"{ctx.image}: SIFT keypoint agreement with cv2 {rec:.3f} / "
                 f"{prec:.3f} is below {AGREE_MIN}")
    kps = [to_cv(k) for k in our_kps(ctx)]
    with cv_mode():
        _, desc = cv2.SIFT_create().compute(ctx.get("half"), kps)
    return np.clip(np.rint(desc), 0, 255).astype(np.uint8)


def rs_points(pts, shape):
    """The forward map of op_rotscale."""
    rows, cols = shape
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([dx * FWD_COS - dy * FWD_SIN + cx, dx * FWD_SIN + dy * FWD_COS + cy], 1)


def cv_rotscale(half, rot):
    """(matches, correct) of cv2.SIFT with the same ratio test."""
    sift = cv2.SIFT_create()
    with cv_mode():
        ka, da = sift.detectAndCompute(half, None)
        kb, db = sift.detectAndCompute(rot, None)
    if da is None or db is None or len(kb) < 2:
        return 0, 0
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if not good:
        return 0, 0
    pa = np.array([ka[m.queryIdx].pt for m in good])
    pb = np.array([kb[m.trainIdx].pt for m in good])
    err = np.linalg.norm(rs_points(pa, half.shape) - pb, axis=1)
    return len(good), int((err < 3).sum())


def check_rotscale(ctx):
    """Returns (ours_matches, ours_correct, cv_matches, cv_correct)."""
    ours = read_values(ctx.dump_dir)[(ctx.image, "sift_rotscale")][2]
    cm, cc = cv_rotscale(ctx.get("half"), ctx.get("rotscale"))
    return int(ours[0]), int(ours[1]), cm, cc


def share(correct, matches):
    return correct / matches if matches else 0.0


def _rotscale(ctx):
    om, oc, cm, cc = check_rotscale(ctx)
    if share(oc, om) < share(cc, cm) - PER_IMAGE_MARGIN:
        sys.exit(f"{ctx.image}: SIFT correct share {oc}/{om} is more than "
                 f"{PER_IMAGE_MARGIN:.0%} below cv2.SIFT {cc}/{cm}")
    return ctx.get("rotscale")  # our own warp: the check above is the test


def _half(ctx):
    gray = ctx.get("gray")
    h, w = gray.shape
    return cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


OPS = {
    "half": (_half, 0, "INTER_AREA, factor 2"),
    "sift_desc": (_sift_desc, 1 if OPTIMIZED else 0,
                  "SIFT.compute at our keypoints; also checks sift_kps"),
    "rotscale": (_rotscale, 0, "our warp; checks sift_rotscale against cv2.SIFT"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from compare_opencv import Context, find_images

    print(f"cv2 {cv2.__version__}, useOptimized={OPTIMIZED}")
    print(f"{'image':<9}{'ours':>6}{'cv2':>6}{'exact':>7}{'cv2 found':>11}{'ours found':>12}"
          f"{'ours match':>16}{'%':>7}{'cv2 match':>16}{'%':>7}")
    tot = [0, 0, 0, 0]
    for image in find_images(args.dump):
        ctx = Context(args.dump, image)
        rec, prec, no, nc, exact = check_kps(ctx)
        r = check_rotscale(ctx)
        tot = [a + b for a, b in zip(tot, r)]
        print(f"{image:<9}{no:>6}{nc:>6}{exact:>7}{rec:>11.3f}{prec:>12.3f}"
              f"{r[1]:>8}/{r[0]:<7}{100 * share(r[1], r[0]):>6.1f}"
              f"{r[3]:>8}/{r[2]:<7}{100 * share(r[3], r[2]):>6.1f}")
    ours, ref = share(tot[1], tot[0]), share(tot[3], tot[2])
    print(f"{'total':<9}{'':>42}{tot[1]:>8}/{tot[0]:<7}{100 * ours:>6.1f}"
          f"{tot[3]:>8}/{tot[2]:<7}{100 * ref:>6.1f}")


if __name__ == "__main__":
    main()
