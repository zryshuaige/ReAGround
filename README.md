# ReAGround: Region-Anchored Two-Stage GUI Grounding for High-Resolution Interfaces

<p align="center">
  <img src="assets/method_comparison.png" width="95%">
</p>

**ReAGround** is a lightweight GUI grounding method: a Qwen2.5-VL-3B backbone,
LoRA fine-tuning (rank 8, &alpha;=16), and a *region-anchored two-stage*
inference protocol in which the model first names the functional **area** of
the target element and then outputs the exact **point** inside the cropped,
upscaled region. Everything runs on a single consumer GPU (one RTX 4090, 2–3 h
per training run).

> Status: code release of an ongoing undergraduate-thesis project.
> Result tables are intentionally left as TBD until the controlled experiments
> finish — no number in this repo is fabricated.

## Current method: group A + region-only GRPO

The current experiment keeps the group-A direct-point adapter and trains only the
**inspection-region** task with GRPO. The same final adapter is called twice at
inference:

```text
screenshot + instruction -> <bbox>[x1,y1,x2,y2]</bbox> (normalized to 0–1000)
                         -> crop once
                         -> A-style direct point inside the crop
```

There is no second model, no multi-candidate generation, and no new point-SFT run.
The GRPO run uses **2,000 prompts × 4 rollouts, one epoch**. Training data are
regenerated with the unchanged run-1 recipe and seed; ScreenSpot is never used
for training or reward tuning.

The four scalar rewards are computed only from the generated box and GT:

1. valid `<bbox>` format (0.1);
2. coverage of the 1.3× context-expanded GT (0.4 × coverage);
3. full context coverage within the area limit (0.5);
4. compactness within that valid range (up to 0.5).

An image-sized box therefore receives at most the first two rewards, not the
containment/compactness bonus. Invalid or non-finite coordinates receive zero.

### Isolated GRPO environment (Linux, one A10 24GB)

Do not install these packages into the run-1 SFT environment. vLLM 0.10.0 pins
torch 2.7.1, while TRL 0.21.0 provides the VLM GRPO/colocate interface used by
`train/train_grpo.py`:

```bash
bash scripts/setup_grpo.sh                 # creates .venv-grpo
source .venv-grpo/bin/activate
python data/build_grpo_dataset.py --samples 2000
python train/train_grpo.py --check-only   # versions, paths, data, TRL config
python train/train_grpo.py --dry-run      # two real updates on 8 records
python train/train_grpo.py                # full 2K run
bash scripts/eval_grpo.sh
```

The pinned TRL version is configured with `vllm_mode="colocate"`, four
generations, short 64-token completions, bf16, SDPA, gradient checkpointing, and
`beta=0` (no reference-model pass). Do not add sleep-mode flags from newer TRL
versions. `--no-vllm` selects the slower HF rollout path. `--dry-run` never
overwrites the final adapter.

## News

- **[2026-10]** Added group-A-initialized region GRPO with layered geometric rewards,
  an isolated TRL/vLLM environment, preflight checks, and two-call inference.
- **[2026-10]** Code release: evaluation harness, data builder, LoRA training
  with section-weighted loss, two-stage inference.

## Legacy run-1 SFT method

| Problem on high-resolution GUIs | ReAGround's answer |
|---|---|
| Tiny targets get visually diluted after the VLM's image resize ("can't see") | **Zoom-in**: anchor a coarse area, crop and let `smart_resize` upscale it, then point |
| Weak GUI structural knowledge ("can't find the right place") | **Region anchor**: the model first judges the functional area (`<region>/<area>`), then points inside it (`<point>`) |

Training is plain SFT on a new output protocol — no architecture change.
Supervision falls only on verifiable facts (region name and area box are
rule-derived from the GT box; the point is the GT center):

```
L = 1 * L_region + 1 * L_point (+ 1 * L_format)
```

where each L is the per-section mean cross-entropy (see `train/dataset.py`).

## Controlled experiment design

Same backbone, same LoRA config, same 8K subset, same eval protocol — one
variable at a time:

| Group | Recipe | Question |
|---|---|---|
| A | 8K + plain LoRA SFT (direct point) | cost of 3x less data |
| B | A + point-anchored zoom (ReGround-style) | value of zoom alone *(planned)* |
| C | A + region CoT, no zoom | value of structure reasoning alone *(planned)* |
| D | A + region CoT + zoom (**ReAGround**) | total gain |
| E | ZonUI-3B official weights (24K) | external reference to surpass |

## Main results (TBD — experiments in progress)

ScreenSpot-Pro / ScreenSpot-v2, point-in-bbox accuracy, split by text/icon.

| Model | Pro avg | Pro text | Pro icon | v2 avg |
|---|---|---|---|---|
| Qwen2.5-VL-3B (base) | TBD | | | |
| ZonUI-3B (group E) | TBD | | | |
| Group A (8K direct SFT) | TBD | | | |
| **ReAGround (group D)** | TBD | | | |

Published references to reproduce *before* any training: ZonUI-3B ≈ 29.0% and
base Qwen2.5-VL-3B ≈ 16.0% on ScreenSpot-Pro. Our harness must match these;
see `scripts/eval_zonui_pro.sh`.

## Installation

Tested: Python 3.9, torch 2.1.2+cu121, `transformers==4.49.0` (newer versions
break with torch 2.1 — do not upgrade blindly).

```bash
pip install -r requirements.txt
```

On a fresh GPU server (e.g. AutoDL), `bash scripts/setup_server.sh` installs
deps and downloads both model weights and both benchmarks (via hf-mirror).
An optional [Dockerfile](docker/Dockerfile) is provided for servers with full
root; rental platforms without a docker daemon should use the pip path.

## Data preparation

Benchmarks (evaluation only, never trained on):

- ScreenSpot-Pro (`likaixin/ScreenSpot-Pro`), 1581 samples
- ScreenSpot-v2 (`HongxinLi/ScreenSpot_v2`), 1272 samples

Training corpora (same public sources as ZonUI-3B):

```bash
python data/download_corpus.py        # ShowUI-web / ShowUI-desktop / UGround (gated)
# AMEX: manual steps printed by the script (https://www.yxchai.com/AMEX)

python data/build_dataset.py --config configs/data_recipe.yaml --fmt twostage
# -> data/outputs/train_twostage.jsonl (16K sequences) + val.jsonl (250 held-out)
```

Sampling: platform-balanced (web/mobile/desktop ≈ 1:1:1), fixed seed, at most
30 entries per unique screenshot (the desktop corpus contains only ~100 unique
screenshots). All area/region/point labels are rule-derived from GT boxes —
no human annotation added. ShowUI-web ships element descriptions, not task
queries; we synthesize the query with a documented template
(`data/build_dataset.py::iter_showui_web`).

## Training

```bash
python train/train_lora.py --config configs/train_lora.yaml          # group D
python train/train_lora.py --config configs/train_lora.yaml --dry-run # 2-step sanity run
```

LoRA rank 8 / &alpha;=16 on attention+MLP projections, bf16, batch 1 ×
grad-accum 48, lr 2e-4, cosine — matching ZonUI-3B's recipe. Only the adapter
is saved.

## Evaluation

```bash
# golden check first: reproduce ZonUI-3B ~29.0% on ScreenSpot-Pro
bash scripts/eval_zonui_pro.sh

# ReAGround, two-stage inference
python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
    --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/reaground_lora \
    --mode twostage --out outputs/group_d_pro.jsonl
```

Per-sample JSONL output feeds directly into error decomposition (hit / near-miss
/ far-miss). `tests/test_core.py` covers parsing, coordinate round-trips, the
section-weighted loss math, and the sequence builders — run it before trusting
any number.

## Repository structure

```
protocol.py             prompts, tags, resize budget, model I/O (single source of truth)
eval/eval_screenspot.py ScreenSpot-v2 / Pro / held-out-set evaluation
infer/parse.py          output parsing + coordinate mapping (self-tested)
infer/pipeline.py       direct and two-stage inference
data/download_corpus.py training-corpus download (HF mirror aware)
data/build_dataset.py   8K sampling + two-stage sequence construction
train/dataset.py        tokenization with per-token loss sections
train/train_lora.py     LoRA SFT + section-weighted loss trainer
configs/                all hyperparameters (data recipe, LoRA, lambdas)
scripts/                server setup + per-group run scripts
tests/                  unit tests (no GPU needed)
docker/                 optional Dockerfile
```

## Acknowledgement

[Qwen2.5-VL](https://arxiv.org/abs/2502.13923),
[ZonUI-3B](https://github.com/Han1018/ZonUI-3B) (baseline recipe and protocol),
[ShowUI](https://github.com/showlab/ShowUI),
[UGround](https://github.com/OSU-NLP-Group/UGround),
[AMEX](https://www.yxchai.com/AMEX),
[ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

## Citation

Thesis manuscript in preparation; citation entry will be added upon publication.

## License

Apache-2.0 (code). The training corpora keep their own licenses — UGround data
is CC BY-NC-SA 4.0 (non-commercial use only).
