#!/usr/bin/env bash
set -euo pipefail

python data/build_grpo_dataset.py --config configs/data_recipe.yaml --samples 2000
python train/train_grpo.py --config configs/grpo_region.yaml --check-only
python train/train_grpo.py --config configs/grpo_region.yaml --dry-run
python train/train_grpo.py --config configs/grpo_region.yaml
