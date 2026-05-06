#!/usr/bin/env bash
# Run on the dev desktop to populate wheels/ for offline laptop install.
# Target laptop must run the same Python (3.12) and same OS family (Ubuntu x86_64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p wheels
pip download \
  -r requirements-lock.txt \
  -d wheels/ \
  --platform manylinux2014_x86_64 \
  --python-version 3.12 \
  --only-binary=:all:
echo "✓ wheels/ populated. Total: $(ls wheels/ | wc -l) files, $(du -sh wheels/ | cut -f1)"
