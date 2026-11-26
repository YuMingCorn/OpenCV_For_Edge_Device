#!/bin/bash
# Fail if any changed C file differs from BASE in anything but comments and
# whitespace at line ends. Usage: tools/check_comment_only.sh [BASE]
# (default origin/main). Needs GCC (clang has no -fpreprocessed); on Windows
# set CC_STRIP="wsl gcc". Spacing inside a line must stay as it was.
set -u -o pipefail
base=${1:-origin/main}
cc=${CC_STRIP:-gcc}
cd "$(git rev-parse --show-toplevel)" || exit 2
strip() {
  tr -d '\r' | $cc -fpreprocessed -dD -E -P -x c - |
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d'
}
probe=$(printf 'int a; /* c */\n  int b; // d\n' | strip | tr '\n' '|')
if [ "$probe" != "int a;|int b;|" ]; then
  echo "check_comment_only: '$cc -fpreprocessed' does not work"
  exit 2
fi
bad=0; n=0
for f in $( { git diff --name-only "$base" -- '*.c' '*.h'
              git ls-files --others --exclude-standard -- '*.c' '*.h'; } | sort -u); do
  n=$((n + 1))
  if [ ! -f "$f" ]; then echo "removed: $f"; bad=1; continue; fi
  if ! git cat-file -e "$base:$f" 2>/dev/null; then echo "new file: $f"; bad=1; continue; fi
  a=$(git show "$base:$f" | strip) || { echo "strip failed: $f"; bad=1; continue; }
  b=$(strip < "$f") || { echo "strip failed: $f"; bad=1; continue; }
  [ -n "$a" ] || { echo "empty output: $f"; bad=1; continue; }
  [ "$a" = "$b" ] || { echo "code changed: $f"; bad=1; }
done
echo "check_comment_only: $n file(s), $([ $bad = 0 ] && echo ok || echo FAIL)"
exit $bad
