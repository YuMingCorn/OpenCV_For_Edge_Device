#!/usr/bin/env python3
"""Static checks for core/. Each rule comes from a bug found in review.

Usage: python3 tools/lint_core.py [--report]
Exit 1 on any violation. --report also prints the comment density table.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
SKIP = {"stb_image.h", "stb_image_write.h"}

# Comment lines / total lines may only go down. Lower a ceiling after a
# cleanup lands; never raise one.
COMMENT_CEILING = {
    ".c": 0.14,
    ".h": 0.75,
}

RULES = [
    (
        "float-half-up",
        # floor(v + 0.5f) rounds 0.49999997f up (v + 0.5f is 1.0f in float)
        # and rounds ties up, not to even as cvRound does.
        re.compile(r"\bfloor\w*\s*\([^;]*?\+\s*0?\.5f?\s*\)"
                   r"|\(\s*u?int\d*_t\s*\)\s*\([^;]*?\+\s*0?\.5f\s*\)"),
        "float rounding with + 0.5: use a round-half-even helper (cvRound)",
    ),
    (
        "unbounded-float-param",
        # p->x > 0.0f alone lets 1e-30f or INFINITY through, which later
        # overflows int conversions or makes NaN.
        re.compile(r"ASSERT\([^;]*->\w+\s*>=?\s*0\.0f\s*\)\s*;"),
        "float parameter checked only for a lower bound: add a finite upper bound",
    ),
]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def comment_lines(text):
    n, inside = 0, False
    for line in text.splitlines():
        s = line.strip()
        if inside:
            n += 1
            inside = "*/" not in s
        elif s.startswith("//"):
            n += 1
        elif s.startswith("/*"):
            n += 1
            inside = "*/" not in s
    return n


def main():
    report = "--report" in sys.argv
    bad = 0
    rows = []
    for path in sorted(CORE.glob("*.[ch]")):
        if path.name in SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        code = strip_comments(text)
        for name, rx, msg in RULES:
            for m in rx.finditer(code):
                line = code.count("\n", 0, m.start()) + 1
                print(f"{path.relative_to(ROOT)}:{line}: [{name}] {msg}")
                bad += 1
        total = max(1, text.count("\n"))
        ratio = comment_lines(text) / total
        rows.append((path.name, ratio))
        limit = COMMENT_CEILING[path.suffix]
        if ratio > limit:
            print(f"core/{path.name}: [comment-density] {ratio:.0%} > "
                  f"{limit:.0%} ceiling")
            bad += 1
    if report:
        for name, ratio in rows:
            print(f"  {name:22s} {ratio:6.1%}")
    print(f"lint_core: {bad} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
