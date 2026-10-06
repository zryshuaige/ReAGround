# -*- coding: utf-8 -*-
"""LoRA SFT with section-weighted loss (ReAGround training entry point).

Config: configs/train_lora.yaml. Single GPU (designed for one RTX 4090).

  python train/train_lora.py --config configs/train_lora.yaml
  python train/train_lora.py --config configs/train_lora.yaml --dry-run  # 2 steps only
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train.dataset import GroundingSFT, collate

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


class SectionalTrainer(Trainer):
    """L = sum_s lambda_s * mean_CE(section s); see train/dataset.py for sections."""

    def __init__(self, *args, section_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_weights = section_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sections = inputs.pop("loss_sections")
        labels = inputs.pop("labels")
        logits = model(**inputs).logits[:, :-1].float()
        labs = labels[:, 1:]
        secs = sections[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             labs.clamp_min(0).reshape(-1),
                             reduction="none").view(labs.shape)
        ce = ce * (labs != -100)
        loss = ce.new_zeros(())
        for sid, w in self.section_weights.items():
            m = secs == sid
            if m.any():
                loss = loss + w * ce[m].sum() / m.sum()
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="run 2 steps on 8 samples to verify the pipeline, then exit")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    torch.manual_seed(cfg["seed"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(cfg["model"])

    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none",
                      target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    ds = GroundingSFT(cfg["data"], processor)
    if args.dry_run:
        ds.records = ds.records[:8]
    print("training sequences: %d" % len(ds))

    weights = {1: cfg["lambdas"]["think"], 2: cfg["lambdas"]["region"],
               3: cfg["lambdas"]["point"], 4: 1.0}
    targs = TrainingArguments(
        output_dir=cfg["output"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0 if args.dry_run else cfg.get("num_workers", 2),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=2 if args.dry_run else -1,
    )
    trainer = SectionalTrainer(
        model=model, args=targs, train_dataset=ds, section_weights=weights,
        data_collator=lambda fs: collate(fs, processor.tokenizer.pad_token_id))
    trainer.train()
    if not args.dry_run:
        model.save_pretrained(cfg["output"])
        processor.save_pretrained(cfg["output"])
        print("adapter saved to", cfg["output"])


if __name__ == "__main__":
    main()
