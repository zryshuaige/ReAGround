#!/usr/bin/env bash
set -euo pipefail

python data/build_dataset.py --config configs/data_recipe.yaml --fmt direct
python train/train_heads.py --config configs/train_heads.yaml --dry-run
python train/train_heads.py --config configs/train_heads.yaml
python eval/eval_screenspot.py --benchmark custom \
    --custom-jsonl data/outputs/val.jsonl --mode heads \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/single_pass_heads \
    --out outputs/single_pass_heads_val.jsonl

echo "Inspect the fixed held-out metrics before running scripts/eval_heads.sh."
