#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv312}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
rm -f "$VENV_DIR/.kinect-ready"
"$VENV_DIR/bin/python" -m pip --disable-pip-version-check install -r requirements.txt
touch "$VENV_DIR/.kinect-ready"

cat <<EOF
Ready.
Run the server with:
  $VENV_DIR/bin/python server.py

If you run server.py with Python 3.14 later, it will automatically re-run itself
with $VENV_DIR/bin/python when this venv exists.
EOF
