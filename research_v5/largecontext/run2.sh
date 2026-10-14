#!/bin/bash
# Run 2: synthetic pretraining, then the same number of fit-split steps as run 1.
set -euo pipefail
cd "$(dirname "$0")/../.."
fit_root="$(.venv/bin/python -c 'from pathlib import Path; print(Path("dataset/assets").resolve().parent)')"
/usr/bin/time -l .venv/bin/python -m research_v5.largecontext.train \
  --manifest research_v5/synthetic/data/manifest.json \
  --data-root research_v5/synthetic/data \
  --checkpoint research_v5/largecontext/runs/run2/pretrain.pt \
  --loss-log research_v5/largecontext/runs/run2/pretrain-loss.json \
  --steps 3200 --batch 2 --seed 1234 --device mps
/usr/bin/time -l .venv/bin/python -m research_v5.largecontext.train \
  --manifest artifacts/splits/fit.json \
  --data-root "$fit_root" \
  --init research_v5/largecontext/runs/run2/pretrain.pt \
  --checkpoint research_v5/largecontext/runs/run2/model.pt \
  --loss-log research_v5/largecontext/runs/run2/finetune-loss.json \
  --steps 3200 --batch 2 --seed 1234 --device mps
