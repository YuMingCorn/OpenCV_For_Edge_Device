"""Compares our SIFT with cv2.SIFT on the edge cases of tests/test_sift_edge.c.

The C test writes each case image and our keypoints with --dump; this script
runs cv2.SIFT_create() (defaults) on the same images and prints, per case,
our count, the cv2 count and the agreement: the share of cv2 keypoints with
one of ours within 1 pixel, size ratio 0.8..1.25 and angle within 10 degrees
(the rule of tools/opencv_ref/sift.py), and whether the keypoints are exactly
equal: x, y, size, angle and response as float32, compared as multisets.
cv2 runs with setUseOptimized(False) unless --optimized is given.

    build/test_sift_edge --dump <dir>
    python3 tools/sift_edge_cv.py --dump <dir>
"""

import argparse
import os

import cv2
import numpy as np


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    fields, pos = [], 0
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    assert fields[0] == b"P5" and fields[3] == b"255"
    cols, rows = int(fields[1]), int(fields[2])
    pixels = np.frombuffer(data[pos + 1:pos + 1 + rows * cols], dtype=np.uint8)
    return pixels.reshape(rows, cols)


def read_kps(path):
    """N x 5 array: x, y, size, angle, response."""
    rows = [list(map(float, line.split()[:5])) for line in open(path)]
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def exact(a, b):
    """True when a and b hold the same float32 keypoints in any order."""
    key = lambda m: sorted(map(tuple, m.astype(np.float32).tolist()))
    return key(a) == key(b)


def agreement(a, b):
    """Share of keypoints in a with one of b within 1 pixel, size ratio
    0.8..1.25 and angle difference below 10 degrees. None when a is empty."""
    if len(a) == 0:
        return None
    if len(b) == 0:
        return 0.0
    hit = 0
    for k in a:
        d = np.abs(k[3] - b[:, 3]) % 360
        d = np.minimum(d, 360 - d)
        ratio = k[2] / b[:, 2]
        ok = ((np.hypot(b[:, 0] - k[0], b[:, 1] - k[1]) <= 1.0)
              & (ratio > 0.8) & (ratio < 1.25) & (d < 10))
        hit += bool(ok.any())
    return hit / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="directory of test_sift_edge --dump")
    ap.add_argument("--list", action="store_true", help="also print the cv2 keypoints of small cases")
    ap.add_argument("--optimized", action="store_true", help="keep cv2's optimized code paths")
    args = ap.parse_args()
    cv2.setUseOptimized(args.optimized)
    print(f"OpenCV {cv2.__version__}, optimized {cv2.useOptimized()}")
    print(f"{'case':<14} {'size':>8} {'ours':>5} {'cv2':>5}  agreement  exact")
    all_exact = True
    for line in open(os.path.join(args.dump, "cases.txt")):
        name, rows, cols, n = line.split()
        ours = read_kps(os.path.join(args.dump, name + "_kps.txt"))
        assert len(ours) == int(n)
        size = f"{rows}x{cols}"
        if int(rows) == 0 or int(cols) == 0:
            print(f"{name:<14} {size:>8} {len(ours):>5} {'-':>5}  cv2 not run (empty image)")
            continue
        img = read_pgm(os.path.join(args.dump, name + ".pgm"))
        try:
            ref = cv2.SIFT_create().detect(img, None)
        except cv2.error as e:
            msg = str(e).strip().splitlines()[-1]
            print(f"{name:<14} {size:>8} {len(ours):>5} {'error':>5}  {msg}")
            continue
        ref = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response]
                        for k in ref]).reshape(-1, 5)
        agr = agreement(ref[:, :4], ours[:, :4])
        text = "n/a" if agr is None else f"{agr * 100:.1f}%"
        same = exact(ours, ref)
        all_exact = all_exact and same
        print(f"{name:<14} {size:>8} {len(ours):>5} {len(ref):>5}  {text:>9}  {'yes' if same else 'NO'}")
        if args.list and 0 < len(ref) <= 12:
            for k in ref:
                print(f"    cv2 x {k[0]:.2f} y {k[1]:.2f} size {k[2]:.2f} angle {k[3]:.1f}")
    print("all exact" if all_exact else "NOT all exact")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
