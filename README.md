# ReAGround: Region-Anchored GUI Grounding

ReAGround grounds a natural-language instruction to a click point on high-resolution GUI screenshots. It uses one Qwen2.5-VL-3B model and one LoRA adapter in two calls: predict an inspection region, crop it once, then predict the click point in the crop.

```text
screenshot + instruction
    -> <bbox>[x1,y1,x2,y2]</bbox>  (normalized to 0–1000)
    -> crop once
    -> (x,y) in the crop
    -> map the point back to the original screenshot
```

There is no second model, multi-candidate search, or benchmark-dependent fallback. Invalid boxes or points count as misses.

> **Status:** ongoing undergraduate-thesis experiments. Accuracy tables remain TBD until the complete controlled runs finish.

## Current training recipe

The same final adapter is trained in three stages:

1. **Group A:** existing direct-point SFT adapter (`outputs/group_a`).
2. **Region SFT warmup:** one epoch over 2,000 region prompts. The target is the deterministic 1.3× GT context box. This stage teaches the box protocol and prevents zero-reward GRPO cold start.
3. **Region GRPO:** the same 2,000 prompts, four rollouts per prompt, one epoch. Point prediction is not retrained with a separate SFT run.

Training examples are regenerated with the unchanged run-1 corpus adapters, sampling recipe, seed, per-image cap, and held-out validation exclusion. ScreenSpot-Pro and ScreenSpot-v2 are evaluation-only.

### Region reward

For predicted box `B`, expand GT by 1.3× to obtain context `G+`, and let `c = area(B ∩ G+) / area(G+)`:

| Component | Reward |
|---|---:|
| Strictly valid `<bbox>` and coordinates | 0.1 |
| Progressive context coverage | 0.4 × c |
| `c >= 0.999` and area within the limit | +0.5 |
| Compactness after successful coverage | +0.0 to 0.5 |

The area limit is `max(area(G+), 25% of the image)`. An image-sized box can receive at most 0.5. Invalid, non-finite, reversed, or out-of-range coordinates receive zero.

## Requirements

- Linux with one A10 24 GB GPU for the current GRPO recipe.
- Python 3.10 for the isolated training environment.
- Base model at `models/Qwen2.5-VL-3B-Instruct`.
- Existing group-A adapter at `outputs/group_a`.
- Training corpora configured in `configs/data_recipe.yaml`.
- Evaluation data at `benchmarks/screenspot_pro` and `benchmarks/screenspot_v2`.

Do not install GRPO dependencies into the legacy SFT environment. The pinned environment is:

```text
torch 2.7.1          torchvision 0.22.1
transformers 4.55.0  trl 0.21.0
vllm 0.10.0          peft 0.17.1
accelerate 1.10.1    datasets 3.6.0
```

Install it with:

```bash
cd ReAGround
bash scripts/setup_grpo.sh
source .venv-grpo/bin/activate
```

## Train on one A10

Run the complete pipeline:

```bash
bash scripts/train_grpo_region.sh
```

This executes, in order:

```bash
python data/build_grpo_dataset.py --samples 2000

python train/train_region_sft.py --config configs/region_sft.yaml --dry-run
python train/train_region_sft.py --config configs/region_sft.yaml

python train/train_grpo.py --config configs/grpo_region.yaml --check-only
python train/train_grpo.py --config configs/grpo_region.yaml --dry-run
python train/train_grpo.py --config configs/grpo_region.yaml
```

Region SFT dry-run requires two non-zero-gradient optimizer steps. GRPO preflight checks every real prompt against the 4,096-token engine limit without truncating image input. GRPO dry-run fails unless it observes non-zero rewards, non-zero within-group reward variance, and non-zero gradients. A run that merely completes two steps is **not** considered valid.

Outputs:

| Path | Contents |
|---|---|
| `data/outputs/train_grpo_region.jsonl` | Shared 2K region-SFT/GRPO records |
| `outputs/group_a_region_sft` | Warmed adapter |
| `outputs/group_a_region_sft_grpo` | Final adapter |

Use fresh output directories for a new experiment. Do not resume from or evaluate the earlier zero-gradient `outputs/group_a_region_grpo` checkpoints.

### A10/vLLM settings

`train/train_grpo.py` applies the TRL-0.21-compatible colocate settings at engine construction:

- vLLM memory utilization `0.40`; `0.35` was measured to leave no usable KV cache on the A10.
- `max_model_len=4096`, with a full-dataset prompt-length assertion.
- Video inputs disabled because this task sends images only; this also avoids the video-profiling rotary failure observed with driver 525.105.17.
- Profiling `max_pixels` aligned with the protocol's `1280 × 28 × 28` image budget.
- bf16, SDPA, gradient checkpointing, 64 completion tokens, `beta=0`, and four generations.

The code pins TRL 0.21 and does not use sleep-mode flags from newer releases. `--no-vllm` selects slower HF rollout for diagnosis, not the recommended A10 run.

## Evaluate

```bash
bash scripts/eval_grpo.sh
```

Equivalent individual commands:

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

Each prediction is saved as JSONL. The metric is point-in-GT-box accuracy; inspect per-sample outputs before reporting aggregate results.

## Results

| Model | ScreenSpot-Pro avg | ScreenSpot-v2 avg |
|---|---:|---:|
| ReAGround region SFT + GRPO | TBD | TBD |

No accuracy improvement is claimed yet. An earlier direct-A-to-GRPO attempt produced only point-style completions: all observed rewards and gradients were zero. That run is a cold-start negative result, not a valid evaluation of GRPO. The warmup pipeline in this revision still requires a complete A10 training and evaluation run.

## Tests

```bash
python tests/test_core.py
python tests/test_grpo.py

# Optional real-tokenizer and sequence-builder checks
REAGROUND_MODEL=models/Qwen2.5-VL-3B-Instruct \
REAGROUND_TEST_IMAGE=/path/to/screenshot.png \
python tests/test_core.py
```

The tests cover coordinate parsing and mapping, reward geometry, SFT targets and loss sections, vLLM argument injection, and dry-run learning-signal gates. CPU tests cannot replace the required A10 end-to-end run.

## Legacy controlled experiments

The original run-1 SFT code remains available for controlled comparisons:

```bash
python train/train_lora.py --config configs/train_group_a.yaml
python train/train_lora.py --config configs/train_lora.yaml
```

That environment uses Python 3.9, torch 2.1.2+cu121, and `transformers==4.49.0`; see `requirements.txt`. Keep it separate from `.venv-grpo`.

## Repository map

```text
protocol.py                    Shared prompts, resize budget, and model I/O
data/build_dataset.py          Run-1 corpus sampling and direct/two-stage labels
data/build_grpo_dataset.py     Shared 2K records and region-SFT box targets
train/dataset.py               Tokenization and answer-only loss masks
train/train_lora.py            Legacy direct/two-stage LoRA SFT
train/train_region_sft.py      Region warmup from group A
train/train_grpo.py            A10 preflight and region GRPO
train/grpo_reward.py           Strict box parsing and geometric rewards
infer/pipeline.py              Direct, legacy two-stage, and GRPO inference
eval/eval_screenspot.py        ScreenSpot-Pro/v2 evaluation
configs/                       Data and training configurations
scripts/                       Setup, training, and evaluation entry points
tests/                         CPU and real-tokenizer tests
```

## Acknowledgements

[Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [ZonUI-3B](https://github.com/Han1018/ZonUI-3B), [ShowUI](https://github.com/showlab/ShowUI), [UGround](https://github.com/OSU-NLP-Group/UGround), [AMEX](https://www.yxchai.com/AMEX), and [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

## License

Apache-2.0 for code. Training corpora retain their own licenses; UGround is CC BY-NC-SA 4.0 (non-commercial use only).
