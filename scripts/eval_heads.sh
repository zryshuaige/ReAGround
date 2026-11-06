#!/usr/bin/env bash
set -euo pipefail

test -f outputs/single_pass_heads/head_config.json
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/single_pass_heads \
    --mode heads --out outputs/single_pass_heads_pro.jsonl
python eval/eval_screenspot.py --benchmark v2 --data-dir benchmarks/screenspot_v2 \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/single_pass_heads \
    --mode heads --out outputs/single_pass_heads_v2.jsonl
