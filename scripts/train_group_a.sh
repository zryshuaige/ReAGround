#!/usr/bin/env bash
# Group A: 8K subset + plain LoRA SFT (direct point output).
set -e
python data/build_dataset.py --config configs/data_recipe.yaml --fmt direct
python train/train_lora.py --config configs/train_group_a.yaml
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/group_a \
    --mode direct --out outputs/group_a_pro.jsonl
python eval/eval_screenspot.py --benchmark v2 --data-dir benchmarks/screenspot_v2 \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/group_a \
    --mode direct --out outputs/group_a_v2.jsonl
