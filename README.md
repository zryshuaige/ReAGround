# ReAGround: Region-Anchored Grounding for High-Resolution GUIs

ReAGround is a two-stage visual grounding framework that localizes GUI elements through **region prediction followed by cropped-region pointing**. It uses a single Qwen2.5-VL-3B backbone and a single LoRA adapter throughout training and inference.

The method combines region-level supervised fine-tuning with Group Relative Policy Optimization (GRPO), using geometric rewards to learn inspection regions that preserve target context while avoiding unnecessary full-image crops.

## Method

Given a screenshot and a natural-language instruction, ReAGround performs two model calls:

```text
Screenshot + instruction
        │
        ▼
Region prediction
<bbox>[x1,y1,x2,y2]</bbox>  (normalized to 0–1000)
        │
        ▼
Crop once
        │
        ▼
Point prediction in the cropped region
(x,y)
        │
        ▼
Map the point to the original screenshot
```

The region stage identifies where interaction should occur; the point stage retains the direct-grounding behavior learned during supervised fine-tuning. Inference uses one model, one crop, and no candidate search or benchmark-dependent fallback.

### Training

The training pipeline contains three stages:

1. **Direct-point SFT.** Learn direct GUI pointing on the run-1 training mixture.
2. **Region SFT warmup.** Train the same adapter on 2,000 region prompts for one epoch. Targets are deterministic 1.3× ground-truth context boxes. This stage establishes a valid region-output distribution before reinforcement learning.
3. **Region GRPO.** Optimize the region policy on the same 2,000 prompts with four rollouts per prompt for one epoch.

All stages share the same backbone and LoRA adapter. Training data are constructed using fixed corpus adapters, sampling rules, random seed, per-image limits, and validation exclusion. ScreenSpot-Pro and ScreenSpot-v2 are used only for evaluation.

### Geometric rewards

Let `B` be the predicted region and `G+` the ground-truth box expanded by 1.3×. Define coverage as

```text
c = area(B ∩ G+) / area(G+)
```

GRPO combines four scalar rewards:

| Component | Reward |
|---|---:|
| Valid `<bbox>` syntax and coordinates | 0.1 |
| Progressive context coverage | 0.4 × c |
| Complete context coverage within the area limit | +0.5 |
| Region compactness after successful coverage | +0.0 to 0.5 |

The maximum permitted area is `max(area(G+), 25% of the image)`. Consequently, an image-sized box receives at most 0.5 and cannot obtain the containment or compactness bonuses.

## Installation

The GRPO pipeline targets Linux with one A10 24 GB GPU and Python 3.10.

```bash
git clone https://github.com/zryshuaige/ReAGround.git
cd ReAGround
bash scripts/setup_grpo.sh
source .venv-grpo/bin/activate
```

Pinned dependencies:

```text
torch 2.7.1          torchvision 0.22.1
transformers 4.55.0  trl 0.21.0
vllm 0.10.0          peft 0.17.1
accelerate 1.10.1    datasets 3.6.0
```

Keep this environment separate from the direct-SFT environment described in `requirements.txt`.

### Required artifacts

```text
models/Qwen2.5-VL-3B-Instruct/   Base model
outputs/group_a/                 Direct-point LoRA adapter
benchmarks/screenspot_pro/       ScreenSpot-Pro
benchmarks/screenspot_v2/        ScreenSpot-v2
```

Training corpora and sampling rules are configured in `configs/data_recipe.yaml`. The corpus download entry point is `data/download_corpus.py`.

## Reproduction

### 1. Build the region dataset

```bash
python data/build_grpo_dataset.py --samples 2000
```

This writes shared region-SFT and GRPO records to `data/outputs/train_grpo_region.jsonl`.

### 2. Train the region policy

```bash
# Region SFT: two-step runtime check, then one epoch
python train/train_region_sft.py --config configs/region_sft.yaml --dry-run
python train/train_region_sft.py --config configs/region_sft.yaml

# GRPO: data/engine preflight, learning-signal check, then one epoch
python train/train_grpo.py --config configs/grpo_region.yaml --check-only
python train/train_grpo.py --config configs/grpo_region.yaml --dry-run
python train/train_grpo.py --config configs/grpo_region.yaml
```

The complete sequence is also available as:

```bash
bash scripts/train_grpo_region.sh
```

The GRPO dry-run requires non-zero rewards, within-group reward variance, and gradients. The preflight checks every processed prompt against the engine's sequence-length budget without truncating the image input.

Output adapters:

```text
outputs/group_a_region_sft/       Region-SFT adapter
outputs/group_a_region_sft_grpo/  Final region-GRPO adapter
```

### 3. Evaluate

```bash
bash scripts/eval_grpo.sh
```

Or run each benchmark separately:

```bash
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
  --model models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/group_a_region_sft_grpo --mode grpo \
  --out outputs/group_a_region_sft_grpo_pro.jsonl

python eval/eval_screenspot.py --benchmark v2 --data-dir benchmarks/screenspot_v2 \
  --model models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/group_a_region_sft_grpo --mode grpo \
  --out outputs/group_a_region_sft_grpo_v2.jsonl
```

Predictions are stored as per-sample JSONL. Accuracy follows the ScreenSpot point-in-ground-truth-box criterion; parsing failures count as misses.

## Implementation details

| Component | Setting |
|---|---|
| Backbone | Qwen2.5-VL-3B-Instruct |
| Adapter | LoRA rank 8, alpha 16 |
| Region prompts | 2,000 |
| GRPO rollouts | 4 per prompt |
| GRPO epochs | 1 |
| KL coefficient | `beta=0` |
| Rollout engine | vLLM colocate |
| Precision | bf16 |
| Attention | SDPA |
| Maximum completion length | 64 tokens |
| vLLM sequence length | 4,096 tokens |
| A10 vLLM memory utilization | 0.40 |

The implementation uses the native Transformers, PEFT, TRL, and vLLM training paths. Image-only profiling disables unused video inputs, and the profiling pixel budget matches the shared inference protocol.

## Results

Results are reported using point-in-box accuracy on the complete evaluation sets.

| Model | ScreenSpot-Pro | ScreenSpot-v2 |
|---|---:|---:|
| ReAGround | TBD | TBD |

## Validation

```bash
python tests/test_core.py
python tests/test_grpo.py
python infer/parse.py
```

Optional real-tokenizer and sequence-builder checks:

```bash
REAGROUND_MODEL=models/Qwen2.5-VL-3B-Instruct \
REAGROUND_TEST_IMAGE=/path/to/screenshot.png \
python tests/test_core.py
```

The test suite covers coordinate transformations, output parsing, reward geometry, supervised targets, loss masks, vLLM engine arguments, and training-signal validation.

## Repository structure

```text
protocol.py                    Shared prompts, resize budget, and model I/O
data/build_dataset.py          Run-1 data sampling and sequence construction
data/build_grpo_dataset.py     Region-SFT and GRPO dataset construction
train/dataset.py               Tokenization and answer-only loss masks
train/train_lora.py            Direct and two-stage SFT baselines
train/train_region_sft.py      Region SFT warmup
train/train_grpo.py            Region GRPO and A10 preflight
train/grpo_reward.py           Box parsing and geometric rewards
infer/pipeline.py              Direct and region-anchored inference
eval/eval_screenspot.py        ScreenSpot-Pro and ScreenSpot-v2 evaluation
configs/                       Data and training configurations
scripts/                       Setup, training, and evaluation entry points
tests/                         Unit and protocol tests
```

## Acknowledgements

ReAGround builds on [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [ZonUI-3B](https://github.com/Han1018/ZonUI-3B), [TRL](https://github.com/huggingface/trl), and [vLLM](https://github.com/vllm-project/vllm). Training and evaluation resources include [ShowUI](https://github.com/showlab/ShowUI), [UGround](https://github.com/OSU-NLP-Group/UGround), [AMEX](https://www.yxchai.com/AMEX), and [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

## License

The code is released under Apache-2.0. Training corpora retain their original licenses; UGround is distributed under CC BY-NC-SA 4.0.
