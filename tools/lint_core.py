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
    ".c": 0.13,
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


# Per-rule skips: rule name -> file names the rule does not scan.
RULE_SKIP = {
}

# Static helpers that ASSERT their Mat argument is not NULL; a call to one
# counts as the check for the mat-null-assert rule.
MAT_VALIDATORS = {"check_u8c1", "check_src", "draw_channels"}


def match_close(code, i, open_ch, close_ch):
    """Index just past the bracket that closes the one at code[i]."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] == open_ch:
            depth += 1
        elif code[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(code)


def split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


FUNC_START = re.compile(r"^[A-Za-z_][^;{}()#=\n]*?\b(\w+)\s*\(", re.M)


def functions(code):
    """(name, is_static, params, body, body_offset) of each definition.

    clang-format puts every top-level declaration at column 0 and indents
    everything inside a body, so a column-0 'name(' starts a top-level item.
    """
    for m in FUNC_START.finditer(code):
        po = m.end() - 1
        pc = match_close(code, po, "(", ")")
        rest = code[pc:]
        brace = len(rest) - len(rest.lstrip())
        if not rest[brace:].startswith("{"):
            continue
        bo = pc + brace
        bc = match_close(code, bo, "{", "}")
        head = code[m.start():po]
        yield (m.group(1), re.match(r"\s*static\b", head) is not None,
               code[po + 1:pc - 1], code[bo:bc], bo)


def assert_spans(body, extra_calls=()):
    names = "ASSERT" + "".join("|" + re.escape(n) for n in extra_calls)
    for m in re.finditer(r"\b(%s)\s*\(" % names, body):
        po = m.end() - 1
        yield m.group(1), m.start(), body[po:match_close(body, po, "(", ")")]


MAT_PARAM = re.compile(r"^(?:const\s+)?struct\s+Mat\s*\*\s*(?:const\s+)?(\w+)$")


def check_mat_null(code):
    """Pointer Mat parameters of public functions must be ASSERTed non-NULL
    before the first dereference (name->x or mat_*(name, ...))."""
    for fname, is_static, params, body, off in functions(code):
        if is_static:
            continue
        for p in split_top(params):
            pm = MAT_PARAM.match(p)
            if not pm:
                continue
            v = re.escape(pm.group(1))
            deref = re.search(r"\b%s\s*->|\bmat_\w+\s*\(\s*%s\b" % (v, v), body)
            if deref is None:
                continue
            checked = len(body)
            for kind, at, args in assert_spans(body, MAT_VALIDATORS):
                if kind == "ASSERT":
                    c = re.search(r"\(\s*%s\s*[)&|]|\b%s\s*!=\s*NULL\b"
                                  r"|\bNULL\s*!=\s*%s\b" % (v, v, v), args)
                    if c:
                        checked = min(checked, at + c.start())
                elif re.match(r"\(\s*%s\s*[,)]" % v, args):
                    checked = min(checked, at)
            if checked > deref.start():
                yield off + deref.start(), (
                    f"{fname}: '{pm.group(1)}' dereferenced before an "
                    "ASSERT that it is not NULL")


EARLY_EMPTY = re.compile(r"\bif\s*\(([^{};]*)\)\s*\{?\s*return\b")


def check_empty_before_data(code):
    """An empty Mat may have data == NULL, so ASSERTs on ->data must come
    after the early return for an empty (or too small) image."""
    for fname, _, _, body, off in functions(code):
        early = None
        for m in EARLY_EMPTY.finditer(body):
            cond = m.group(1)
            if (re.search(r"->rows\s*(==|<=?)", cond)
                    and re.search(r"->cols\s*(==|<=?)", cond)):
                early = m.start()
                break
        if early is None:
            continue
        for _, at, args in assert_spans(body):
            if at < early and re.search(r"->data\s*[!=]=|[!=]=\s*[^&|]*->data",
                                        args):
                yield off + at, (f"{fname}: ASSERT on ->data before the "
                                 "empty-image return")


INT_DECL = re.compile(r"(?<![\w])int\s+([^;(){}]*)")


def check_int_plus_step(code):
    """'i + 16 <= n' with an int i overflows when n is near INT_MAX; write
    'n - i >= 16' or use size_t."""
    for fname, _, params, body, off in functions(code):
        ints = set()
        for m in INT_DECL.finditer(params + ";" + body):
            for part in split_top(m.group(1)):
                d = re.match(r"\**\s*(\w+)", part)
                if d:
                    ints.add(d.group(1))
        for m in re.finditer(r"\bfor\s*\([^;]*;\s*(\w+)\s*\+\s*\w+\s*<=", body):
            if m.group(1) in ints:
                yield off + m.start(), (
                    f"{fname}: int '{m.group(1)}' + step <= bound can "
                    "overflow: use 'bound - i >= step' or size_t")


FUNC_RULES = [
    ("mat-null-assert", check_mat_null),
    ("empty-before-data-assert", check_empty_before_data),
    ("int-plus-step-bound", check_int_plus_step),
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
        rel = path.relative_to(ROOT).as_posix()
        for name, rx, msg in RULES:
            if path.name in RULE_SKIP.get(name, ()):
                continue
            for m in rx.finditer(code):
                line = code.count("\n", 0, m.start()) + 1
                print(f"{rel}:{line}: [{name}] {msg}")
                bad += 1
        if path.suffix == ".c":
            for name, fn in FUNC_RULES:
                if path.name in RULE_SKIP.get(name, ()):
                    continue
                for at, msg in fn(code):
                    line = code.count("\n", 0, at) + 1
                    print(f"{rel}:{line}: [{name}] {msg}")
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
