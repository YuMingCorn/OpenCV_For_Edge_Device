#!/usr/bin/env python3
"""Check that the architecture docs match the code.

Usage: python3 tools/check_arch_doc.py
Exit 1 on any mismatch. Checks:
- every C source in CMakeLists.txt is named in figure 1 of docs/README.md,
  figure 1 names no other C file, and the float box and the improc box list
  the same files as CMake;
- every module appears in the diagram under "## 架構" in README.md;
- the internal-header table in docs/README.md lists exactly the core/*.c
  files that include each internal header (a core header not in improc.h);
- every core/, tools/ and tests/ path named in README.md and docs/*.md exists,
  and so does every relative link or image target in them;
- every Kodak compare loop (for m in ...) lists every tests/kodak_ops_*.c;
- every manual build loop (for f in ...; core/$f.c) and every ESP-IDF SRCS
  list names every improc_core source;
- README.md keeps its build and Kodak loops, docs/kodak-results.md its Kodak
  loop.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
DOC = ROOT / "docs" / "README.md"
README = ROOT / "README.md"


def read(p):
    return p.read_text(encoding="utf-8").replace("\r", "")


def mermaid_after(text, heading):
    """The first Mermaid block in the section that starts at heading."""
    at = text.find(heading)
    if at < 0:
        return None
    end = text.find("\n## ", at + 1)
    section = text[at:end if end >= 0 else len(text)]
    m = re.search(r"^```mermaid\n(.*?)^```", section, re.S | re.M)
    return m.group(1) if m else None


def cmake_groups():
    """C files of improc_core, of the float sources, and of improc itself."""
    text = re.sub(r"#.*", "", read(ROOT / "CMakeLists.txt"))
    groups = {}
    for var in ("IMPROC_CORE_SOURCES", "IMPROC_FLOAT_SOURCES"):
        m = re.search(r"set\(" + var + r"\s(.*?)\)", text, re.S)
        if m is None:
            sys.exit(f"check_arch_doc: {var} not found in CMakeLists.txt")
        groups[var] = set(re.findall(r"core/(\w+\.c)", m.group(1)))
    m = re.search(r"add_library\(improc STATIC([^)]*)\)", text)
    groups["improc"] = set(re.findall(r"core/(\w+\.c)", m.group(1) if m else ""))
    return groups


def node_files(fig, node):
    """C files named in one Mermaid node or subgraph line, e.g. float_ops[...]."""
    m = re.search(r"^\s*(?:subgraph )?" + node + r"\b.*$", fig, re.M)
    return set(re.findall(r"\b(\w+\.c)\b", m.group(0))) if m else None


def includes(path):
    return set(re.findall(r'^#include "([\w.]+)"', read(path), re.M))


def main():
    errors = []
    doc, readme = read(DOC), read(README)
    groups = cmake_groups()
    sources = set().union(*groups.values())
    if "mat_io.c" not in groups["improc"]:
        errors.append("CMakeLists.txt: mat_io.c not found in add_library(improc)")

    fig1 = mermaid_after(doc, "## 1.") or ""
    named = set(re.findall(r"\b(\w+\.c)\b", fig1))
    for f in sorted(sources - named):
        errors.append(f"docs/README.md figure 1: {f} is missing")
    for f in sorted(named - sources):
        errors.append(f"docs/README.md figure 1: {f} is not in CMakeLists.txt")
    # Files must also sit in the right box of figure 1.
    for node, var in (("float_ops", "IMPROC_FLOAT_SOURCES"),
                      ("mat_io_c", "improc")):
        got = node_files(fig1, node)
        if got != groups[var]:
            errors.append(f"docs/README.md figure 1: {node} lists "
                          f"{sorted(got or [])}, CMakeLists.txt has "
                          f"{sorted(groups[var])}")

    top = mermaid_after(readme, "## 架構") or ""
    for f in sorted(sources):
        if not re.search(r"\b" + re.escape(f[:-2]) + r"\b", top):
            errors.append(f"README.md 架構 diagram: module {f[:-2]} is missing")

    public = includes(CORE / "improc.h")
    internal = {h.name for h in CORE.glob("*.h")
                if h.name not in public and h.name != "improc.h"
                and not h.name.startswith("stb_")}
    rows = {}
    table = re.search(r"^\| 內部標頭 \|.*?\n\n", doc, re.S | re.M)
    for line in (table.group(0) if table else "").splitlines():
        m = re.match(r"\| `([\w.]+\.h)` \|.*\|([^|]*)\|\s*$", line)
        if m is None:
            continue
        if m.group(1) not in internal:
            errors.append(f"docs/README.md: internal-header row for "
                          f"{m.group(1)}, which is not an internal header")
        rows[m.group(1)] = set(re.findall(r"`(\w+\.c)`", m.group(2)))
    users = {h: set() for h in internal}
    for c in CORE.glob("*.c"):
        for h in includes(c) & internal:
            users[h].add(c.name)
    for h in sorted(internal):
        if h not in rows:
            errors.append(f"docs/README.md: no internal-header row for {h}")
        elif rows[h] != users[h]:
            errors.append(f"docs/README.md: {h} row lists {sorted(rows[h])}, "
                          f"code has {sorted(users[h])}")

    stems = sorted(p.stem[len("kodak_ops_"):]
                   for p in (ROOT / "tests").glob("kodak_ops_*.c"))
    loops = {}
    for md in [README, *sorted((ROOT / "docs").glob("*.md"))]:
        rel = md.relative_to(ROOT).as_posix()
        text = read(md)
        # A repo path starts the token (not inside ~/x/tools/...); a token with
        # a shell variable or a pattern ($m, *, <module>, {a,b}) is skipped.
        # ASCII only: \w would also take the CJK text right after a path.
        for m in re.finditer(
                r"(?<![A-Za-z0-9_/.~-])((?:core|tools|tests)/[A-Za-z0-9_./$*<>{},-]*)",
                text):
            path = m.group(1).rstrip(".,")
            if re.search(r"[$*<>{}]", path):
                continue
            if not (ROOT / path).exists():
                errors.append(f"{rel}: path {path} does not exist")
        # Relative links and images, e.g. [x](porting.md#a) or ![x](02-mat.svg).
        # Code blocks are skipped: C code like tbl[i](arg) is not a link.
        prose = re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.S | re.M)
        for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)"
                             r"|<img [^>]*src=\"([^\"]+)\"", prose):
            link = (m.group(1) or m.group(2)).split("#")[0]
            if not link or re.match(r"[a-z]+:", link, re.I):
                continue
            if not (md.parent / link).exists():
                errors.append(f"{rel}: link {link} does not exist")
        core_c = sorted(f[:-2] for f in sources if f != "mat_io.c")
        loops[rel] = [0, 0]
        for m in re.finditer(r"^for f in ([\w ]+); do\n.*core/\$f\.c", text, re.M):
            loops[rel][0] += 1
            if sorted(m.group(1).split()) != core_c:
                errors.append(f"{rel}: core build loop lists {m.group(1).split()}, "
                              f"improc_core has {core_c}")
        # ESP-IDF component: SRCS core/a.c core/b.c ... INCLUDE_DIRS
        for m in re.finditer(r"\bSRCS\s(.*?)\bINCLUDE_DIRS", text, re.S):
            listed = sorted(re.findall(r"core/(\w+)\.c", m.group(1)))
            if listed != core_c:
                errors.append(f"{rel}: SRCS lists {listed}, "
                              f"improc_core has {core_c}")
        for m in re.finditer(r"^for m in ([\w ]+); do", text, re.M):
            loops[rel][1] += 1
            if sorted(m.group(1).split()) != stems:
                errors.append(f"{rel}: Kodak loop lists {m.group(1).split()}, "
                              f"tests have {stems}")
    # A removed or reworded loop would otherwise pass silently.
    for rel, kind, i in (("README.md", "core build", 0), ("README.md", "Kodak", 1),
                         ("docs/kodak-results.md", "Kodak", 1)):
        if loops.get(rel, [0, 0])[i] == 0:
            errors.append(f"{rel}: no {kind} loop found")

    for e in errors:
        print(e)
    print(f"check_arch_doc: {len(sources)} sources, {len(internal)} internal "
          f"headers, {len(stems)} Kodak modules, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
