#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT_DIR/checkpoints" "$ROOT_DIR/datasets"

cd "$ROOT_DIR/data"

if [ ! -d chartqa_vcot ]; then
  unzip -q train_chartqa_vcot.zip
  find chartqa_vcot -name '.DS_Store' -delete
fi

if [ -d ChartQA ] && [ ! -f "$ROOT_DIR/datasets/train_full.parquet" ]; then
  python preprocess.py
else
  echo "Prepared chartqa_vcot; run data/preprocess_data.sh later for ChartQA images."
fi
