# ReAGround: Region-Anchored Grounding for High-Resolution GUIs

ReAGround uses two independent LoRA adapters on one Qwen2.5-VL-3B backbone:

- **R (region adapter):** predicts an inspection box on the full screenshot.
- **A (direct adapter):** the original direct-SFT adapter, frozen and used only to predict a point inside the crop.

The current implementation uses adaptive context targets and position-aware GRPO rewards. R is trained from the original base model, so region training cannot overwrite A's direct-point behavior.

## Method

```text
Screenshot + instruction
        │
        ▼
R: <bbox>[x1,y1,x2,y2]</bbox>  (normalized to 0–1000)
        │
        ▼
Crop once
        │
        ▼
Frozen A: (x,y) in the cropped region
        │
        ▼
Map the point to the original screenshot
```

One backbone is loaded with two named adapters. Inference activates `region` for the first call and `direct` for the second. Points outside the resized crop are rejected and stored as misses rather than being mapped outside the selected region.

## Experimental single-pass heads

`modeling_reaground.py` also implements an untrained experimental alternative to external cropping. It adds a `Linear(2048, 1)` Region Head after LLM layer 8 and a `Linear(2048, 3)` Point Head after layer 36. Region logits are added to the point-selection scores; the other two Point Head outputs predict a continuous offset inside the selected visual-token cell.

This path performs one forward pass without text generation, cropping, adapter switching, or token deletion. Its prompt places the instruction before the image so that causal visual states can attend to the instruction. It does not claim lower latency, and the new heads require training before evaluation.

```bash
# Rebuild the unchanged 8K direct sample with original GT bbox fields.
python data/build_dataset.py --config configs/data_recipe.yaml --fmt direct
python train/train_heads.py --config configs/train_heads.yaml --dry-run
python train/train_heads.py --config configs/train_heads.yaml

python eval/eval_screenspot.py --benchmark custom \
  --custom-jsonl data/outputs/val.jsonl --mode heads \
  --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/single_pass_heads \
  --out outputs/single_pass_heads_val.jsonl
```

The joint loss is point-cell cross-entropy plus within-cell Smooth L1 plus balanced region BCE. Evaluation reports final point-in-box accuracy and region/point center-cell diagnostics separately. ScreenSpot-Pro/v2 remain evaluation-only and must not be used to select the region layer or loss weights.

## Training

### 1. Direct adapter A

A is the existing direct-point LoRA from run 1. It is not modified by the new region experiments.

### 2. Region SFT warmup

R starts from a new rank-8, alpha-16 LoRA on the original Qwen2.5-VL-3B base. It trains for one epoch on the same 2,000 prompts used by GRPO.

Adaptive targets contain the ground-truth box while avoiding the very small crops produced by a uniform 1.3× expansion. Small targets receive a context region of approximately 10% of image area. The preferred area range is 10%–25%, but target boxes may exceed it when necessary to contain large or elongated targets.

### 3. Region GRPO

GRPO uses four rollouts per prompt for one epoch. It has four separately logged components:

| Component | Reward |
|---|---:|
| Valid `<bbox>` syntax and coordinates | 0.1 |
| Continuous distance between predicted and GT centers | 0 to 0.4 |
| Fraction of the GT box covered by the prediction | 0 to 0.5 |
| Area score after the GT center is inside the prediction | 0 to 0.5 |

The area component is zero when the GT center is outside the predicted box. A correctly located box scores best around 10%–25% of image area; larger targets are allowed when the adaptive target itself exceeds that range. An off-target box cannot earn a high score merely by having an acceptable area.

The continuous position component provides a learning signal even when the prediction and GT do not overlap.

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

Keep this environment separate from the original direct-SFT environment in `requirements.txt`.

### Required artifacts

```text
models/Qwen2.5-VL-3B-Instruct/   Base model
outputs/group_a/                 Frozen direct-point A adapter
corpus/raw/                      Original training corpora
data/outputs/val.jsonl           The fixed 250-sample held-out set
benchmarks/screenspot_pro/       ScreenSpot-Pro, evaluation only
benchmarks/screenspot_v2/        ScreenSpot-v2, evaluation only
```

If `val.jsonl` is absent, create it with the original sampling recipe:

```bash
python data/build_dataset.py --config configs/data_recipe.yaml --fmt direct
```

## Reproduction

### 1. Build adaptive data and train R with SFT

```bash
bash scripts/train_region_r_sft.sh
```

This builds `data/outputs/train_region_adaptive.jsonl`, runs the two-step check, trains `outputs/region_r_sft`, and evaluates R + frozen A on the fixed held-out set.

**Inspect the held-out region metrics before starting GRPO.** A non-zero gradient dry-run is only a runtime check, not evidence that region localization has been learned. Do not continue if held-out region coverage shows no meaningful learning.

### 2. Train R with GRPO

```bash
bash scripts/train_region_r_grpo.sh
```

This runs preflight, a two-update learning-signal check, formal GRPO, and another held-out evaluation. The final adapter is `outputs/region_r_grpo`.

The legacy single-adapter scripts and configs remain available for reproducing earlier experiments; they are not the current method.

### 3. Evaluate on ScreenSpot

```bash
bash scripts/eval_region_r.sh
```

Or run one benchmark directly:

```bash
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
  --model models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/region_r_grpo --direct-adapter outputs/group_a --mode dual \
  --out outputs/region_r_grpo_pro.jsonl
```

Evaluation prints final point accuracy, parse failures, region GT-center coverage, and region GT overlap. Pro/v2 are not used to select area ranges, rewards, or checkpoints.

## Implementation details

| Component | Setting |
|---|---|
| Backbone | One Qwen2.5-VL-3B-Instruct |
| Region adapter R | New LoRA, rank 8, alpha 16 |
| Direct adapter A | Original direct-SFT LoRA, frozen |
| Region prompts | 2,000 |
| GRPO rollouts | 4 per prompt |
| Epochs | SFT 1, GRPO 1 |
| Preferred region area | 10%–25%, with target-containment exceptions |
| KL coefficient | `beta=0` |
| Rollout engine | vLLM colocate |
| Precision / attention | bf16 / SDPA |
| Maximum completion length | 64 tokens |
| vLLM sequence length | 4,096 tokens |

## Results

All values are measured point-in-box accuracy. TBD means that no result is recorded for that method and benchmark.

| Method | ScreenSpot-Pro | ScreenSpot-v2 |
|---|---:|---:|
| Run-1 A direct SFT | 20.7% | 85.8% |
| Legacy shared-LoRA region SFT + GRPO | 5.0% (79/1581) | TBD |
| Current independent R + frozen A | TBD | TBD |
| Experimental single-pass heads | TBD | TBD |

The legacy 5.0% result is retained as a completed negative result. Neither the adaptive-reward dual-adapter method nor the single-pass heads have benchmark results.

## Validation

```bash
python tests/test_core.py
python tests/test_grpo.py
python tests/test_heads.py
python infer/parse.py
```

The tests cover legacy rewards, adaptive target construction, continuous position rewards, coordinate mapping, adapter switching, dense-head targets and losses, continuous point decoding, and region-to-point fusion.

Optional real-model checks:

```bash
REAGROUND_MODEL=models/Qwen2.5-VL-3B-Instruct \
REAGROUND_TEST_IMAGE=/path/to/screenshot.png \
python tests/test_core.py

REAGROUND_MODEL=models/Qwen2.5-VL-3B-Instruct \
REAGROUND_TEST_IMAGE=/path/to/screenshot.png \
python tests/test_heads.py
```

The head test additionally performs a real forward/backward pass and checks that PEFT saves and restores both heads with the LoRA adapter.

## Repository structure

```text
protocol.py                    Shared prompts, resize budget, model loading
modeling_reaground.py          Experimental single-pass Region/Point Heads
data/build_grpo_dataset.py     Legacy and adaptive region dataset construction
train/train_region_sft.py      Base- or adapter-initialized region SFT
train/train_grpo.py            Legacy and adaptive region GRPO
train/train_heads.py           Joint region and point SFT
train/grpo_reward.py           Legacy rewards and adaptive position/coverage/area rewards
infer/pipeline.py              Direct, dual-adapter, and single-pass inference
eval/eval_screenspot.py        Benchmark and held-out evaluation with region diagnostics
configs/region_r_sft.yaml      Current base-initialized R warmup
configs/grpo_region_r.yaml     Current adaptive-reward GRPO
configs/train_heads.yaml       Experimental single-pass head training
scripts/train_region_r_sft.sh  Data, warmup, and held-out quality check
scripts/train_region_r_grpo.sh GRPO and held-out evaluation
scripts/eval_region_r.sh       Full Pro/v2 evaluation
scripts/train_heads.sh         Head data, dry-run, SFT, and held-out evaluation
scripts/eval_heads.sh          Single-pass Pro/v2 evaluation
```

## Acknowledgements

ReAGround builds on [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [ZonUI-3B](https://github.com/Han1018/ZonUI-3B), [TRL](https://github.com/huggingface/trl), and [vLLM](https://github.com/vllm-project/vllm). Training and evaluation resources include [ShowUI](https://github.com/showlab/ShowUI), [UGround](https://github.com/OSU-NLP-Group/UGround), [AMEX](https://www.yxchai.com/AMEX), and [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

## License

The code is released under Apache-2.0. Training corpora retain their original licenses; UGround is distributed under CC BY-NC-SA 4.0.
