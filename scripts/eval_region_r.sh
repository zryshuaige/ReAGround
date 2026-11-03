#!/usr/bin/env bash
set -euo pipefail

python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct \
    --adapter outputs/region_r_grpo --direct-adapter outputs/group_a --mode dual \
    --out outputs/region_r_grpo_pro.jsonl
python eval/eval_screenspot.py --benchmark v2 --data-dir benchmarks/screenspot_v2 \
    --model models/Qwen2.5-VL-3B-Instruct \
    --adapter outputs/region_r_grpo --direct-adapter outputs/group_a --mode dual \
    --out outputs/region_r_grpo_v2.jsonl
