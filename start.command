#!/usr/bin/env bash
# macOS / Linux: double-click this file to set up and start Argus.
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3 python py; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' 2>/dev/null; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9+ is required but was not found."
  echo "Install it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close..." _
  exit 1
fi
"$PY" argus.py start "$@"
STATUS=$?
if [ $STATUS -ne 0 ]; then read -r -p "Press Enter to close..." _; fi
exit $STATUS
