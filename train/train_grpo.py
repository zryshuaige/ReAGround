# -*- coding: utf-8 -*-
"""Region GRPO from the region-SFT adapter (TRL 0.21 / Qwen2.5-VL)."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from protocol import MAX_PIXELS, build_messages, resized_dims
from train.grpo_reward import REWARD_FUNCS, parse_bbox

MAX_COMPLETION_LENGTH = 64

VERSIONS = {
    "torch": "2.7.1", "transformers": "4.55.0", "trl": "0.21.0",
    "peft": "0.17.1", "accelerate": "1.10.1", "datasets": "3.6.0",
    "vllm": "0.10.0",
}


def check_versions(use_vllm):
    problems = []
    for package, expected in VERSIONS.items():
        if package == "vllm" and not use_vllm:
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append("%s is not installed (need %s)" % (package, expected))
            continue
        if actual != expected and not (package == "torch" and actual.startswith(expected + "+")):
            problems.append("%s==%s, need %s" % (package, actual, expected))
    if problems:
        raise RuntimeError("GRPO environment mismatch:\n  " + "\n  ".join(problems) +
                           "\nUse requirements-grpo.txt in a separate Linux environment.")


def vllm_engine_kwargs(kwargs, cfg):
    patched = dict(kwargs)
    if patched.get("max_model_len") is None:
        patched["max_model_len"] = cfg["max_model_len"]
    limits = dict(patched.get("limit_mm_per_prompt") or {})
    limits["video"] = 0
    patched["limit_mm_per_prompt"] = limits
    processor_kwargs = dict(patched.get("mm_processor_kwargs") or {})
    processor_kwargs.setdefault("max_pixels", MAX_PIXELS)
    patched["mm_processor_kwargs"] = processor_kwargs
    return patched


@contextmanager
def patch_vllm_engine_args(cfg):
    from trl.trainer import grpo_trainer
    original = grpo_trainer.LLM

    def wrapped(*args, **kwargs):
        return original(*args, **vllm_engine_kwargs(kwargs, cfg))

    grpo_trainer.LLM = wrapped
    try:
        yield
    finally:
        grpo_trainer.LLM = original


def validate_dry_run(log_history):
    steps = [entry for entry in log_history if "grad_norm" in entry]
    problems = []
    if len(steps) != 2:
        problems.append("expected 2 optimizer steps, got %d" % len(steps))
    if not any(entry.get("reward", 0.0) > 0 for entry in steps):
        problems.append("all rewards are zero")
    if not any(entry.get("frac_reward_zero_std", 1.0) < 1 for entry in steps):
        problems.append("every prompt group has zero reward variance")
    if not any(entry.get("grad_norm", 0.0) > 0 for entry in steps):
        problems.append("all gradient norms are zero")
    if problems:
        raise RuntimeError("GRPO dry-run has no learning signal: " + "; ".join(problems))


def validate_prompt_budget(longest, max_model_len):
    if longest[0] + MAX_COMPLETION_LENGTH > max_model_len:
        raise ValueError("prompt %s has %d tokens; with %d completion tokens exceeds max_model_len=%d" %
                         (longest[1], longest[0], MAX_COMPLETION_LENGTH, max_model_len))


def check_prompt_lengths(records, processor, max_model_len):
    from PIL import Image
    longest = (0, None)
    for record in records:
        prompt = processor.apply_chat_template(
            build_messages("region", record["instruction"]),
            tokenize=False, add_generation_prompt=True)
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            image = image.resize(resized_dims(*image.size))
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        if length > longest[0]:
            longest = (length, record["id"])
    validate_prompt_budget(longest, max_model_len)
    return longest


def load_records(path, limit=None):
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("empty dataset: " + path)
    for record in records:
        if parse_bbox("<bbox>[%s]</bbox>" % ",".join(map(str, record["gt_bbox"]))) is None:
            raise ValueError("invalid normalized GT box in %s: %s" % (record["id"], record["gt_bbox"]))
        if not os.path.isfile(record["image"]):
            raise FileNotFoundError(record["image"])
    return records


def make_dataset(records, processor):
    from datasets import Dataset, Image as HFImage
    rows = []
    for record in records:
        prompt = processor.apply_chat_template(
            build_messages("region", record["instruction"]),
            tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "image": record["image"],
                     "gt_bbox": record["gt_bbox"]})
    dataset = Dataset.from_list(rows).cast_column("image", HFImage(decode=True))

    def prepare(batch):
        batch["image"] = [image.convert("RGB").resize(resized_dims(*image.size))
                          for image in batch["image"]]
        return batch

    return dataset.with_transform(prepare)


def make_args(cfg, dry_run, no_vllm):
    from trl import GRPOConfig
    use_vllm = cfg["use_vllm"] and not no_vllm
    return GRPOConfig(
        output_dir=cfg["output"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=4,
        num_train_epochs=1,
        max_steps=2 if dry_run else -1,
        learning_rate=cfg["learning_rate"],
        beta=0.0,
        epsilon=0.2,
        loss_type="bnpo",
        scale_rewards=True,
        max_prompt_length=None,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=1.0,
        top_p=1.0,
        mask_truncated_completions=True,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        vllm_tensor_parallel_size=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=4,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_region.yaml")
    parser.add_argument("--dry-run", action="store_true", help="two real GRPO updates on 8 records")
    parser.add_argument("--no-vllm", action="store_true", help="use HF rollout instead of vLLM")
    parser.add_argument("--check-only", action="store_true", help="validate environment, files and config only")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    use_vllm = cfg["use_vllm"] and not args.no_vllm
    check_versions(use_vllm)
    for key in ("model", "adapter", "data"):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError("%s: %s" % (key, cfg[key]))
    records = load_records(cfg["data"], 8 if args.dry_run else None)
    training_args = make_args(cfg, args.dry_run, args.no_vllm)
    if args.check_only:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["model"])
        longest, longest_id = check_prompt_lengths(records, processor, cfg["max_model_len"])
        print("GRPO preflight OK: %d records, max_prompt=%d tokens (%s), output=%s, vLLM=%s" %
              (len(records), longest, longest_id, cfg["output"], use_vllm))
        return

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOTrainer

    torch.manual_seed(cfg["seed"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, cfg["adapter"], is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(cfg["model"])
    processor.tokenizer.padding_side = "left"
    context = patch_vllm_engine_args(cfg) if use_vllm else nullcontext()
    with context:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            reward_funcs=REWARD_FUNCS,
            train_dataset=make_dataset(records, processor),
            processing_class=processor,
        )
    trainer.train()
    if args.dry_run:
        validate_dry_run(trainer.state.log_history)
        print("GRPO dry-run OK: non-zero rewards, reward variance and gradients observed")
    else:
        trainer.save_model(cfg["output"])
        processor.save_pretrained(cfg["output"])
        print("adapter saved to", cfg["output"])


if __name__ == "__main__":
    main()
