#!/usr/bin/env bash
# Sets up the local venv (first run only) and launches GPU Control Panel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=.venv
if [ ! -d "$VENV" ]; then
    echo "Setting up virtualenv in $VENV ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install -r requirements.txt -q
fi

exec "$VENV/bin/python" main.py "$@"
