#!/usr/bin/env bash
set -euo pipefail

python3.10 -m venv .venv-grpo
source .venv-grpo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-grpo.txt
python - <<'PY'
import torch
import trl
import transformers
import vllm
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "trl": trl.__version__, "vllm": vllm.__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY
