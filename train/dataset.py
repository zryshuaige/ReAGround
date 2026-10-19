# -*- coding: utf-8 -*-
"""Training dataset: JSONL records -> tokenized samples with per-token loss sections.

Section ids (loss_sections, aligned with labels):
  0 = prompt / padding            (never supervised)
  2 = <region>..</area> block     (weight lambda_region)
  3 = <point>..</point> or direct "(x, y)"  (weight lambda_point)
  4 = format tokens: <region></region> tags, <|im_end|>  (weight always 1; the model
      must learn the protocol tokens)

Loss (per micro-batch, per section mean then weighted sum):
  L = lambda_region * CE_2 + lambda_point * CE_3 + 1.0 * CE_4
Per-section averaging prevents one long segment from dominating.

Char spans are mapped to tokens via the fast tokenizer's offset mapping, so
segment boundaries can never drift from the tokenization.
"""
import json

import torch
from torch.utils.data import Dataset
from PIL import Image

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from protocol import build_messages, resized_dims

def _spans(full):
    """Char spans (start, end, section_id) covering a target string exactly.
    Section ids: 2=region+area, 3=point, 4=format tokens."""
    spans = []
    pos = 0
    if full.startswith("<region>", pos):
        pos_end = full.index("</area>", pos) + len("</area>")
        spans.append((pos, pos_end, 2))
        pos = pos_end
    if full.startswith("<point>", pos):
        pos_end = full.index("</point>", pos) + len("</point>")
        spans.append((pos, pos_end, 3))
        pos = pos_end
    elif not full.startswith("<|im_end|>", pos):  # direct mode: bare "(x, y)"
        pos_end = full.index("<|im_end|>", pos)
        spans.append((pos, pos_end, 3))
        pos = pos_end
    spans.append((pos, len(full), 4))  # <|im_end|>\n
    return spans


def tokenize_target(tokenizer, target):
    """target string -> (input_ids, section_ids) tensors. Fast tokenizer required."""
    assert tokenizer.is_fast, "need a fast tokenizer for offset mapping"
    full = target + "<|im_end|>\n"
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    spans = _spans(full)
    sec = torch.zeros(len(enc["input_ids"]), dtype=torch.long)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        for cs, ce, sid in spans:
            if cs <= s < ce:
                sec[i] = sid
                break
    return torch.tensor(enc["input_ids"]), sec


class GroundingSFT(Dataset):
    """Reads train_*.jsonl written by data/build_dataset.py."""

    def __init__(self, jsonl_path, processor):
        self.records = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        image = Image.open(rec["image"]).convert("RGB")
        if "crop" in rec:
            image = image.crop(rec["crop"])
        rw, rh = resized_dims(image.width, image.height)
        image = image.resize((rw, rh))
        prompt_text = self.processor.apply_chat_template(
            build_messages(rec["mode"], rec["instruction"]),
            tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        tgt_ids, tgt_secs = tokenize_target(self.tokenizer, rec["target"])
        prompt_ids = enc["input_ids"][0]
        return {
            "input_ids": torch.cat([prompt_ids, tgt_ids]),
            "labels": torch.cat([torch.full_like(prompt_ids, -100), tgt_ids]),
            "loss_sections": torch.cat([torch.zeros_like(prompt_ids), tgt_secs]),
            "attention_mask": torch.ones(len(prompt_ids) + len(tgt_ids),
                                         dtype=torch.long),
            "pixel_values": enc["pixel_values"],  # flat [num_patches, dim]; no batch dim
            "image_grid_thw": enc["image_grid_thw"][0],
        }


def collate(features, pad_id):
    n = max(len(f["input_ids"]) for f in features)
    batch = {"pixel_values": torch.cat([f["pixel_values"] for f in features]),
             "image_grid_thw": torch.stack([f["image_grid_thw"] for f in features])}
    for key, pad in (("input_ids", pad_id), ("labels", -100),
                     ("loss_sections", 0), ("attention_mask", 0)):
        batch[key] = torch.stack([
            torch.cat([f[key], torch.full((n - len(f[key]),), pad,
                                          dtype=f[key].dtype)])
            for f in features])
    return batch
