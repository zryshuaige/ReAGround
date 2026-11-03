# -*- coding: utf-8 -*-
"""Warm up an inspection-box adapter before region GRPO."""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train.dataset import GroundingSFT, collate
from train.grpo_reward import parse_bbox
from train.train_grpo import check_versions, load_records
from train.train_lora import LORA_TARGETS, SectionalTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/region_sft.yaml")
    parser.add_argument("--dry-run", action="store_true", help="two optimizer steps on eight records")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    check_versions(use_vllm=False)
    for key in ("model", "data"):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError("%s: %s" % (key, cfg[key]))
    adapter = cfg.get("adapter")
    if adapter and not os.path.exists(adapter):
        raise FileNotFoundError("adapter: " + adapter)
    records = load_records(cfg["data"], 8 if args.dry_run else None)
    for record in records:
        if record.get("mode") != "region" or parse_bbox(record.get("target", "")) is None:
            raise ValueError("invalid region SFT target; rebuild data with data/build_grpo_dataset.py")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments

    torch.manual_seed(cfg["seed"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        lora = cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora.get("r", 8), lora_alpha=lora.get("alpha", 16),
            lora_dropout=lora.get("dropout", 0.05), bias="none",
            target_modules=LORA_TARGETS)
        model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(cfg["model"])
    dataset = GroundingSFT(cfg["data"], processor)
    if args.dry_run:
        dataset.records = dataset.records[:8]

    training_args = TrainingArguments(
        output_dir=cfg["output"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        max_steps=2 if args.dry_run else -1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1 if args.dry_run else 5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0 if args.dry_run else cfg.get("num_workers", 2),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SectionalTrainer(
        model=model, args=training_args, train_dataset=dataset,
        section_weights={2: 1.0, 4: 1.0},
        data_collator=lambda features: collate(features, processor.tokenizer.pad_token_id))
    trainer.train()
    if args.dry_run:
        steps = [entry for entry in trainer.state.log_history if "grad_norm" in entry]
        if len(steps) != 2 or not all(entry["grad_norm"] > 0 for entry in steps):
            raise RuntimeError("region SFT dry-run did not produce two non-zero gradient steps")
        print("region SFT dry-run OK: two non-zero gradient steps")
    else:
        model.save_pretrained(cfg["output"])
        processor.save_pretrained(cfg["output"])
        print("adapter saved to", cfg["output"])


if __name__ == "__main__":
    main()
