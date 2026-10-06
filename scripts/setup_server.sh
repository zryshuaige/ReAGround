#!/usr/bin/env bash
# One-shot server setup: deps + model weights + benchmarks.
# Usage: bash scripts/setup_server.sh
set -e
export HF_ENDPOINT=https://hf-mirror.com   # domestic servers; comment out overseas

pip install -r requirements.txt

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="models/Qwen2.5-VL-3B-Instruct")
snapshot_download("zonghanHZH/ZonUI-3B", local_dir="models/ZonUI-3B",
                  ignore_patterns=["*.bin"])  # safetensors only, halves the download
snapshot_download("likaixin/ScreenSpot-Pro", repo_type="dataset",
                  local_dir="benchmarks/screenspot_pro",
                  ignore_patterns=[".DS_Store", "**/.DS_Store", ".gitattributes"])
snapshot_download("HongxinLi/ScreenSpot_v2", repo_type="dataset",
                  local_dir="benchmarks/screenspot_v2")
print("models + benchmarks ready")
PY

echo "Training corpora: python data/download_corpus.py"
echo "  - UGround (osunlp/UGround-V1-Data-Box) is GATED: accept terms on HF and huggingface-cli login first"
echo "  - AMEX needs manual steps, see data/download_corpus.py"
