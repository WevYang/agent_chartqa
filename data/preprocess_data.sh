#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/data"

if [ ! -d chartqa_vcot ]; then
  unzip -q train_chartqa_vcot.zip
  find chartqa_vcot -name '.DS_Store' -delete
fi

if [ ! -d ChartQA ]; then
  wget -c -O ChartQA.zip "https://huggingface.co/datasets/ReFocus/ReFocus_Data/resolve/main/images/ChartQA.zip?download=true"
  unzip -q ChartQA.zip
  find ChartQA -name '.DS_Store' -delete
fi

python preprocess.py
