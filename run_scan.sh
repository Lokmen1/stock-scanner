#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

source .venv/bin/activate
set -a
source .env
set +a

python main.py backfill --days 1
python main.py universe
python main.py scan
