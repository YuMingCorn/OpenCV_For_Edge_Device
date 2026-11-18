#!/usr/bin/env python3
"""Compare improc with OpenCV on exhaustive and random inputs, not only Kodak.

    python3 tools/sweep_opencv.py --probe build/improc_probe [--suite NAME]...

Each suite builds its inputs, runs them through improc_probe (tools/
improc_probe.c) and through cv2, and prints the samples, the largest absolute
difference and the share of samples that differ. A suite fails when the share
exceeds its limit, and the script then exits with status 1.

Colour suites use every input value: all 2^24 RGB triples, all 2^16 565 words,
all 256^3 HSV triples. Resize suites use random images at random sizes, with a
fixed seed, so a run is repeatable.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

SEED = 20261118



def hsv_vector_width():
    """Pixels per vector step of OpenCV's 8-bit HSV2RGB on this machine.

    The vector loop truncates and the scalar loop rounds, and HSV (0, 1, 1)
    gives 0 truncated and 1 rounded. A row of w such pixels is all vector
    work exactly when w is a multiple of the step, so the step is the
    smallest w whose last pixel comes out truncated. 32 with AVX2 builds of
    opencv-python 4.10.0; 16 is expected with 128-bit vectors only.
    """
    for w in (4, 8, 16, 32, 64, 128):
        row = np.zeros((1, w, 3), np.uint8)
        row[..., 1:] = 1
        if cv2.cvtColor(row, cv2.COLOR_HSV2RGB)[0, -1, 1] == 0:
            return w
    return 1 << 30  # no vector loop: every pixel is scalar


class Probe:
    def __init__(self, exe, work):
        self.exe = exe
        self.work = work

    def run(self, img, op):
        """img: uint8 HxWxC or int16 HxWxC array. Returns the output array."""
        if img.ndim == 2:
            img = img[..., None]
        rows, cols, cn = img.shape
        kind = "u8" if img.dtype == np.uint8 else "s16"
        src = os.path.join(self.work, "in.raw")
        dst = os.path.join(self.work, "out.raw")
        np.ascontiguousarray(img).tofile(src)
        cmd = [self.exe, src, str(rows), str(cols), f"{kind}c{cn}", dst] + [str(a) for a in op]
        subprocess.run(cmd, check=True)
        out = np.fromfile(dst, dtype=np.uint8)
        if op[0] == "resize":
            return out.reshape(int(op[2]), int(op[1]), cn)
        return out.reshape(rows, cols, -1)


def rgb_cube():
    v = np.arange(1 << 24, dtype=np.uint32)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8).reshape(4096, 4096, 3)


def hsv_cube():
    h, s, v = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")
    return np.stack([h, s, v], -1).astype(np.uint8).reshape(4096, 4096, 3)


def as3(a):
    return a if a.ndim == 3 else a[..., None]


class Tally:
    def __init__(self):
        self.n = 0
        self.diff = 0
        self.worst = 0

    def add(self, ours, ref):
        ours = as3(ours).astype(np.int64)
        ref = as3(ref).astype(np.int64)
        assert ours.shape == ref.shape, (ours.shape, ref.shape)
        d = np.abs(ours - ref)
        self.n += d.size
        self.diff += int((d != 0).sum())
        self.worst = max(self.worst, int(d.max()) if d.size else 0)

    def pct(self):
        return 100.0 * self.diff / self.n if self.n else 0.0


# ---- colour ------------------------------------------------------------------

_SWAP = [0, 2, 1]  # improc YUV is Y U V; OpenCV YCrCb is Y Cr Cb


def suite_gray(p):
    cube = rgb_cube()
    rng = np.random.default_rng(SEED)
    rgba = np.concatenate([cube, rng.integers(0, 256, cube.shape[:2] + (1,), dtype=np.uint8)], -1)
    for code, cv_code, img in [("RGB2GRAY", cv2.COLOR_RGB2GRAY, cube),
                               ("BGR2GRAY", cv2.COLOR_BGR2GRAY, cube),
                               ("RGBA2GRAY", cv2.COLOR_RGBA2GRAY, rgba),
                               ("BGRA2GRAY", cv2.COLOR_BGRA2GRAY, rgba)]:
        t = Tally()
        t.add(p.run(img, ["cvt", code]), cv2.cvtColor(img, cv_code))
        yield code, "all 2^24 RGB", t


def suite_gray565(p):
    w = np.arange(65536, dtype=np.uint32)
    le = np.stack([w & 255, w >> 8], -1).astype(np.uint8).reshape(256, 256, 2)
    ref = cv2.cvtColor(le, cv2.COLOR_BGR5652GRAY)
    t = Tally()
    t.add(p.run(le, ["cvt", "BGR5652GRAY"]), ref)
    yield "BGR5652GRAY", "all 2^16 words", t
    t = Tally()
    t.add(p.run(le[..., ::-1], ["cvt", "BGR565BE2GRAY"]), ref)
    yield "BGR565BE2GRAY", "all 2^16 words", t


def suite_yuv(p):
    cube = rgb_cube()
    for code, cv_code in [("RGB2YUV", cv2.COLOR_RGB2YCrCb), ("BGR2YUV", cv2.COLOR_BGR2YCrCb)]:
        t = Tally()
        t.add(p.run(cube, ["cvt", code]), cv2.cvtColor(cube, cv_code)[..., _SWAP])
        yield code, "all 2^24 RGB", t
    for code, cv_code in [("YUV2RGB", cv2.COLOR_YCrCb2RGB), ("YUV2BGR", cv2.COLOR_YCrCb2BGR)]:
        t = Tally()
        t.add(p.run(cube, ["cvt", code]), cv2.cvtColor(np.ascontiguousarray(cube[..., _SWAP]), cv_code))
        yield code, "all 2^24 YUV", t


def suite_hsv(p):
    cube = hsv_cube()
    for code, cv_code in [("HSV2RGB", cv2.COLOR_HSV2RGB), ("HSV2BGR", cv2.COLOR_HSV2BGR)]:
        t = Tally()
        t.add(p.run(cube, ["cvt", code]), cv2.cvtColor(cube, cv_code))
        yield code, "all 256^3 HSV, width 4096", t
    hsv = cv2.cvtColor(rgb_cube(), cv2.COLOR_RGB2HSV)
    t = Tally()
    t.add(p.run(hsv, ["cvt", "HSV2RGB"]), cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    yield "HSV2RGB", "RGB2HSV of all 2^24 RGB", t
    # OpenCV runs each row through a vector loop of `step` pixels and the
    # last (width % step) pixels through a scalar loop that rounds
    # instead of truncating. improc follows the vector loop at every column,
    # so the two parts are reported apart.
    rng = np.random.default_rng(SEED)
    step = hsv_vector_width()
    vec, tail = Tally(), Tally()
    for w in range(1, 101):
        img = rng.integers(0, 256, (64, w, 3), dtype=np.uint8)
        img[..., 0] %= 180
        ours, ref = p.run(img, ["cvt", "HSV2RGB"]), cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        cut = w // step * step
        vec.add(ours[:, :cut], ref[:, :cut])
        tail.add(ours[:, cut:], ref[:, cut:])
    yield "HSV2RGB", f"widths 1-100, vector columns ({step})", vec
    yield "HSV2RGB", "widths 1-100, OpenCV scalar tail", tail


# ---- convert_scale -----------------------------------------------------------

_ALPHAS = list(range(-1024, 1025, 7)) + [128, 256, 384, 640]
_BETAS = [-300, -10, 0, 10, 37, 200]


def suite_scale(p):
    x = np.arange(256, dtype=np.uint8).reshape(16, 16)
    t = Tally()
    for a in _ALPHAS:
        for b in _BETAS:
            # addWeighted with a zero second weight is convertTo(alpha, beta):
            # saturate, no absolute value, like convert_scale.
            t.add(p.run(x, ["scale", a, b]), cv2.addWeighted(x, a / 256, x, 0, b))
    yield "convert_scale", f"all 256 u8 x {len(_ALPHAS) * len(_BETAS)} (alpha, beta)", t


def suite_scale_abs(p):
    x = np.arange(-32768, 32768, dtype=np.int16).reshape(256, 256)
    t = Tally()
    for a in [1, 3, 7, 64, 128, 255, 256, 257, 384, 1000, -256, -384]:
        for b in [-100, 0, 10, 255]:
            t.add(p.run(x, ["scale_abs", a, b]), cv2.convertScaleAbs(x, alpha=a / 256, beta=b))
    yield "convert_scale_abs", "all 2^16 s16 x 48 (alpha, beta)", t


# ---- resize ------------------------------------------------------------------


def _sizes(rng, n, lo, hi):
    return [tuple(rng.integers(lo, hi, 4).tolist()) for _ in range(n)]


def _resize_case(p, t, rng, sw, sh, dw, dh, interp, cv_interp):
    cn = int(rng.choice([1, 3, 4]))
    img = rng.integers(0, 256, (sh, sw, cn), dtype=np.uint8)
    ref = cv2.resize(img if cn > 1 else img[..., 0], (dw, dh), interpolation=cv_interp)
    t.add(p.run(img, ["resize", dw, dh, interp]), ref)


def suite_linear(p):
    rng = np.random.default_rng(SEED)
    groups = {
        "random sizes 1-300": _sizes(rng, 80, 1, 300),
        "Kodak sizes": [(768, 512, 329, 214), (512, 768, 219, 321), (768, 512, 1152, 768),
                        (768, 512, 256, 170), (768, 512, 1000, 300)],
        "integer upscale x2-x4": [(w, h, w * k, h * k) for w, h, k in
                                  [(17, 9, 2), (40, 31, 3), (25, 64, 4), (3, 5, 2)]],
        # OpenCV rounds the source position to float32, whose step grows with
        # the size; small images never show that.
        "large sizes": [(4000, 3000, 3333, 2222), (2000, 1500, 1111, 1111),
                        (1024, 1024, 777, 777), (1920, 1080, 1280, 720),
                        (3000, 2000, 4321, 2345), (4096, 64, 4095, 63)],
    }
    for name, cases in groups.items():
        t = Tally()
        for sw, sh, dw, dh in cases:
            _resize_case(p, t, rng, sw, sh, dw, dh, "linear", cv2.INTER_LINEAR)
        yield "INTER_LINEAR", name, t


def suite_area(p):
    rng = np.random.default_rng(SEED + 1)
    ratios = [(2, 2), (2, 1), (1, 2), (3, 3), (4, 4), (2, 3), (4, 2), (5, 3),
              (6, 6), (7, 3), (8, 8), (5, 7)]
    groups = {
        "integer ratios": [(dw * kx, dh * ky, dw, dh) for kx, ky in ratios
                           for dw, dh in rng.integers(1, 60, (4, 2)).tolist()],
        "non-integer downscale": [(sw, sh, int(rng.integers(1, sw + 1)), int(rng.integers(1, sh + 1)))
                                  for sw, sh in [tuple(int(v) for v in rng.integers(2, 250, 2))
                                                 for _ in range(60)]],
        "Kodak sizes": [(768, 512, 384, 256), (768, 512, 256, 170), (768, 512, 192, 128),
                        (768, 512, 329, 214), (512, 768, 170, 256)],
        # Integer ratios other than 2x2 go through OpenCV's float
        # sum * (1 / area); with larger areas that can round differently
        # from the exact mean.
        "integer ratios 4x7 to 11x13": [(32, 105, 8, 15), (60, 70, 6, 7), (90, 72, 10, 8),
                                        (128, 96, 16, 12), (143, 169, 13, 13)],
    }
    for name, cases in groups.items():
        t = Tally()
        for sw, sh, dw, dh in cases:
            _resize_case(p, t, rng, sw, sh, dw, dh, "area", cv2.INTER_AREA)
        yield "INTER_AREA", name, t
    # Upscaling under INTER_AREA is a separate OpenCV path (linear weights).
    t = Tally()
    for sw, sh in [tuple(int(v) for v in rng.integers(1, 80, 2)) for _ in range(30)]:
        _resize_case(p, t, rng, sw, sh, sw * 3 // 2 + int(rng.integers(0, 3)),
                     sh * 3 // 2 + int(rng.integers(0, 3)), "area", cv2.INTER_AREA)
    yield "INTER_AREA", "upscale about 1.5x", t
    t = Tally()
    for sw, sh in [tuple(rng.integers(1, 150, 2).tolist()) for _ in range(40)]:
        _resize_case(p, t, rng, sw, sh, sw + int(rng.integers(1, 150)),
                     sh + int(rng.integers(1, 150)), "area", cv2.INTER_AREA)
    yield "INTER_AREA", "upscale, random sizes", t
    # Sizes with a large gcd put many destination cells on exact source
    # boundaries, where OpenCV's double floor can pick the previous pixel.
    t = Tally()
    for sw, sh, dw, dh in [(800, 600, 1280, 960), (640, 480, 1000, 750),
                           (1024, 768, 1280, 960), (720, 480, 1080, 720),
                           (1000, 1000, 3000, 2000)]:
        _resize_case(p, t, rng, sw, sh, dw, dh, "area", cv2.INTER_AREA)
    yield "INTER_AREA", "upscale, large gcd", t
    # One axis grows and the other shrinks. OpenCV finds the source pixel
    # with a double floor(dx * sw / dw); where that product is an exact
    # integer, rounding can land one pixel lower, and the whole column then
    # differs. improc uses the exact integer.
    t = Tally()
    for sw, sh in [tuple(rng.integers(2, 150, 2).tolist()) for _ in range(40)]:
        dw, dh = sw + int(rng.integers(1, 150)), int(rng.integers(1, sh))
        if rng.integers(0, 2):
            sw, sh, dw, dh = sh, sw, dh, dw
        _resize_case(p, t, rng, sw, sh, dw, dh, "area", cv2.INTER_AREA)
    yield "INTER_AREA", "one axis up, one down", t
    t = Tally()
    for sw, sh in [(512, 768), (768, 512), (40, 60), (2, 2)]:
        _resize_case(p, t, rng, sw, sh, sw * 3 // 2, sh * 3 // 2, "area", cv2.INTER_AREA)
    yield "INTER_AREA", "upscale exactly 1.5x", t


SUITES = {
    "gray": suite_gray,
    "gray565": suite_gray565,
    "yuv": suite_yuv,
    "scale": suite_scale,
    "scale_abs": suite_scale_abs,
    "linear": suite_linear,
    "area": suite_area,
    "hsv": suite_hsv,
}

# Largest share of differing samples (percent) per op. An op that is not
# listed is reported only; so is any row whose inputs name a key of REPORT_ONLY.
# Exact ops must stay at 0. The resize limits allow for OpenCV's float
# coordinates, which improc computes exactly (see docs/kodak-results.md).
LIMITS = {
    "RGB2GRAY": 0, "BGR2GRAY": 0, "RGBA2GRAY": 0, "BGRA2GRAY": 0,
    "BGR5652GRAY": 0, "BGR565BE2GRAY": 0,
    "RGB2YUV": 0, "BGR2YUV": 0, "YUV2RGB": 0, "YUV2BGR": 0,
    "convert_scale": 0, "convert_scale_abs": 0,
    "HSV2RGB": 0, "HSV2BGR": 0,
    "INTER_LINEAR": 1, "INTER_AREA": 1,
}
REPORT_ONLY = ["scalar tail"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", required=True, help="path to the improc_probe executable")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES), help="run only these suites")
    ap.add_argument("--work", help="directory for the raw files (default: a new temporary directory)")
    args = ap.parse_args()
    work = args.work or tempfile.mkdtemp(prefix="improc_sweep_")
    os.makedirs(work, exist_ok=True)
    probe = Probe(os.path.abspath(args.probe), work)
    failed = 0
    print(f"OpenCV {cv2.__version__}")
    print(f"{'op':<18} {'inputs':<34} {'samples':>11} {'max':>4} {'differ %':>9} {'limit %':>8}")
    for name in args.suite or list(SUITES):
        for op, inputs, t in SUITES[name](probe):
            limit = None if any(k in inputs for k in REPORT_ONLY) else LIMITS.get(op)
            bad = limit is not None and t.pct() > limit
            failed += bad
            lim = "-" if limit is None else f"{limit:g}"
            print(f"{op:<18} {inputs:<34} {t.n:>11} {t.worst:>4} {t.pct():>9.4f} {lim:>8}"
                  + ("  FAIL" if bad else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
