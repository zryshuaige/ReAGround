#!/usr/bin/env bash
# Base model reference on ScreenSpot-Pro (leaderboard reports ~16.0%).
set -e
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct --mode direct \
    --out outputs/base_pro.jsonl
