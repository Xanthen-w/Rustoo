#!/usr/bin/env bash
# Start the local research app (http://localhost:8501). Local-only, no telemetry.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
    echo "Installing app dependencies into .venv ..."
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt -r requirements-app.txt
fi
exec .venv/bin/streamlit run app/app.py
