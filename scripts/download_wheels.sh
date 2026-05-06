#!/usr/bin/env bash
# Run on the dev desktop to populate wheels/ for offline laptop install.
# Target laptop must match the dev desktop's Python (3.12) and OS family
# (Ubuntu x86_64) — wheels downloaded here are tagged for the running
# interpreter and OS (see PEP 600 manylinux tags).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p wheels
# Prefer wheels matching the dev box's interpreter/OS. We avoid the strict
# --platform / --python-version flags because some packages (e.g. contourpy
# 1.3.3) only publish manylinux_2_28 wheels — the lock file pin would then
# fail to resolve under --platform manylinux2014. The dev box and demo laptop
# are both Ubuntu 22.04+ x86_64 with Python 3.12, so the natively-resolved
# wheels are compatible across both.
pip download \
  -r requirements-lock.txt \
  -d wheels/ \
  --only-binary=:all:
echo "✓ wheels/ populated. Total: $(ls wheels/ | wc -l) files, $(du -sh wheels/ | cut -f1)"
