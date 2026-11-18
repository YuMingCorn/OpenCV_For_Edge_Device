#!/usr/bin/env python3
"""Prints improc_orb_pattern, the 256 BRIEF test pairs of core/features2d.c.

Integer only, so the C test (tests/test_features.c) regenerates the same
table: xorshift32 seeded with 0x9E3779B9; a coordinate is the sum of four
(state >> 27) values minus 62, divided by 3 and rounded half away from zero
(about Gaussian, sigma 6.2); a pair (x1, y1, x2, y2) is rejected when a
coordinate is outside -13..13 or both points are the same.

    python3 tools/gen_orb_pattern.py
"""

SEED = 0x9E3779B9
PAIRS = 256
LIMIT = 13


def xorshift32(s):
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s


def pattern():
    state, pairs = SEED, []
    while len(pairs) < PAIRS:
        pair = []
        for _ in range(4):
            total = 0
            for _ in range(4):
                state = xorshift32(state)
                total += state >> 27
            c = total - 62
            q = (abs(c) + 1) // 3
            pair.append(q if c >= 0 else -q)
        if max(abs(v) for v in pair) <= LIMIT and pair[:2] != pair[2:]:
            pairs.append(pair)
    return pairs


def main():
    print("const signed char improc_orb_pattern[256][4] = {")
    rows = pattern()
    for i in range(0, PAIRS, 4):
        print("    " + " ".join("{" + ", ".join(map(str, r)) + "},"
                                 for r in rows[i:i + 4]))
    print("};")


if __name__ == "__main__":
    main()
