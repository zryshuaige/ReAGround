#!/usr/bin/env bash
# Group D: 8K subset + LoRA + region-anchored two-stage (full ReAGround).
set -e
python data/build_dataset.py --config configs/data_recipe.yaml --fmt twostage
python train/train_lora.py --config configs/train_lora.yaml
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/reaground_lora \
    --mode twostage --out outputs/group_d_pro.jsonl
python eval/eval_screenspot.py --benchmark v2 --data-dir benchmarks/screenspot_v2 \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/reaground_lora \
    --mode twostage --out outputs/group_d_v2.jsonl
