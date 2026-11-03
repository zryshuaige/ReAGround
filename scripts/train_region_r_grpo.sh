#!/usr/bin/env bash
set -euo pipefail

python train/train_grpo.py --config configs/grpo_region_r.yaml --check-only
python train/train_grpo.py --config configs/grpo_region_r.yaml --dry-run
python train/train_grpo.py --config configs/grpo_region_r.yaml

python eval/eval_screenspot.py --benchmark custom --custom-jsonl data/outputs/val.jsonl \
    --model models/Qwen2.5-VL-3B-Instruct \
    --adapter outputs/region_r_grpo --direct-adapter outputs/group_a --mode dual \
    --out outputs/region_r_grpo_val.jsonl
