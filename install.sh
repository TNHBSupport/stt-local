#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Install complete."
echo "Run: source .venv/bin/activate && uvicorn server:app --host 127.0.0.1 --port 9000"
