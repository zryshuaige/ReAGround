# -*- coding: utf-8 -*-
"""Joint SFT for single-pass region and point heads."""
import argparse
import json
import os
import sys

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modeling_reaground import ReAGroundHeadModel, head_loss
from train.dataset import GroundingHeadSFT, collate_heads
from train.train_lora import LORA_TARGETS


class HeadTrainer(Trainer):
    def __init__(self, *args, lambdas=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambdas = lambdas

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        gt_bboxes = inputs.pop("gt_bbox")
        outputs = model(**inputs)
        loss, metrics = head_loss(
            outputs, gt_bboxes, lambda_region=self.lambdas["region"],
            lambda_offset=self.lambdas["offset"])
        self.log({"region_loss": metrics["region"],
                  "point_ce": metrics["point_ce"],
                  "offset_loss": metrics["offset"]})
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_heads.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="two optimizer steps on eight samples")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for key in ("model", "data"):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError("%s: %s" % (key, cfg[key]))

    torch.manual_seed(cfg["seed"])
    model = ReAGroundHeadModel.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn", "sdpa"))
    model.region_layer = cfg["region_layer"]
    model.config.region_layer = cfg["region_layer"]
    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], bias="none",
        target_modules=LORA_TARGETS,
        modules_to_save=["region_head", "point_head"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(cfg["model"])
    dataset = GroundingHeadSFT(cfg["data"], processor)
    if args.dry_run:
        dataset.records = dataset.records[:8]

    training_args = TrainingArguments(
        output_dir=cfg["output"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=1 if args.dry_run else cfg["grad_accum"],
        learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"],
        max_steps=2 if args.dry_run else -1,
        lr_scheduler_type="cosine", bf16=True,
        logging_steps=1 if args.dry_run else 5,
        save_strategy="steps", save_steps=50, save_total_limit=2,
        report_to=[], seed=cfg["seed"], remove_unused_columns=False,
        dataloader_num_workers=0 if args.dry_run else cfg.get("num_workers", 2),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = HeadTrainer(
        model=model, args=training_args, train_dataset=dataset,
        lambdas=cfg["lambdas"],
        data_collator=lambda features: collate_heads(
            features, processor.tokenizer.pad_token_id))
    trainer.train()
    if args.dry_run:
        steps = [entry for entry in trainer.state.log_history if "grad_norm" in entry]
        if len(steps) != 2 or not all(entry["grad_norm"] > 0 for entry in steps):
            raise RuntimeError("head SFT dry-run did not produce two non-zero gradients")
        print("head SFT dry-run OK: two non-zero gradient steps")
    else:
        model.save_pretrained(cfg["output"])
        processor.save_pretrained(cfg["output"])
        with open(os.path.join(cfg["output"], "head_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"region_layer": cfg["region_layer"]}, f, indent=2)
        print("adapter and heads saved to", cfg["output"])


if __name__ == "__main__":
    main()
