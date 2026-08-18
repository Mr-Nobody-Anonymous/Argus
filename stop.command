#!/usr/bin/env bash
# macOS / Linux: double-click this file to stop Argus.
cd "$(dirname "$0")" || exit 1
for c in python3 python py; do
  if command -v "$c" >/dev/null 2>&1; then "$c" argus.py stop; exit $?; fi
done
echo "Python not found."
read -r -p "Press Enter to close..." _
