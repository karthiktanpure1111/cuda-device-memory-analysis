#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p data/output bin
if [[ ! -x bin/box_filter ]]; then
  make
fi
./bin/box_filter \
  --input "${1:-data/input}" \
  --output "${2:-data/output}" \
  --kernel "${3:-5}" \
  --streams "${4:-4}" | tee data/output/run.log
