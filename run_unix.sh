#!/usr/bin/env bash
# PosterProof runner (Unix: macOS / Linux / WSL).
# If you are on Windows, double-click `run_windows.bat` instead; this .sh file
# will not run on stock Windows (no bash by default). If you want to run .sh
# on Windows install WSL or Git Bash.

set -euo pipefail

cd "$(dirname "$0")"

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: Python >= 3.10 not found on this machine." >&2
    echo "Install Python 3.10+ via your package manager or https://www.python.org/downloads/" >&2
    exit 1
fi

echo "Using Python: $("$PY" --version 2>&1)"

VENV_DIR=".venv-unix"
if [ ! -x "$VENV_DIR/bin/python" ] || \
   [ ! -f "$VENV_DIR/bin/activate" ] || \
   ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PY" -m venv --clear "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -r requirements.txt
if [ -f "requirements-dev.txt" ]; then
    echo "Installing dev dependencies ..."
    python -m pip install -r requirements-dev.txt
fi

echo
echo "Running test suite ..."
python -m pytest -v

echo
echo "Starting PosterProof on http://127.0.0.1:8765/ ..."
echo "Press Ctrl+C to stop."
exec python app.py
