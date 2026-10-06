#!/usr/bin/env bash
# Golden check: reproduce ZonUI-3B on ScreenSpot-Pro (paper reports ~29.0%).
# If this does not reproduce, the eval harness has a bug -- fix before training.
set -e
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/ZonUI-3B --mode direct \
    --out outputs/zonui_pro.jsonl
