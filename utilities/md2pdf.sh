#!/usr/bin/env bash
# Convert a markdown file to PDF via pandoc + xelatex.
# Usage: ./md2pdf.sh yourfile.md [output.pdf]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 yourfile.md [output.pdf]" >&2
  exit 1
fi

export PATH="/Library/TeX/texbin:$PATH"

SRC="$1"
DIR="$(cd "$(dirname "$SRC")" && pwd)"
BASE="$(basename "$SRC")"
OUT="${2:-${BASE%.md}.pdf}"

cd "$DIR"

pandoc "$BASE" -o "$OUT" --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="Helvetica Neue" \
  -V monofont="Menlo" \
  -V monofontoptions="Scale=0.85" \
  --toc

echo "Wrote $DIR/$OUT"
