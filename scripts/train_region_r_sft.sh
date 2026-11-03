#!/usr/bin/env bash
set -euo pipefail

python data/build_grpo_dataset.py --config configs/data_recipe.yaml --samples 2000 \
    --target-mode adaptive --min-area-ratio 0.10 --max-area-ratio 0.25 \
    --output data/outputs/train_region_adaptive.jsonl

test -f data/outputs/val.jsonl || {
    echo "missing data/outputs/val.jsonl; run python data/build_dataset.py --fmt direct first" >&2
    exit 1
}

python train/train_region_sft.py --config configs/region_r_sft.yaml --dry-run
python train/train_region_sft.py --config configs/region_r_sft.yaml

python eval/eval_screenspot.py --benchmark custom --custom-jsonl data/outputs/val.jsonl \
    --model models/Qwen2.5-VL-3B-Instruct \
    --adapter outputs/region_r_sft --direct-adapter outputs/group_a --mode dual \
    --out outputs/region_r_sft_val.jsonl

echo "Inspect held-out region coverage before running scripts/train_region_r_grpo.sh."
