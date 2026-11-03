# -*- coding: utf-8 -*-
"""Build region-GRPO records from the exact run-1 sampling recipe."""
import argparse
import json
import math
import os
import sys

import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_dataset import ADAPTERS, sample_balanced
from train.grpo_reward import expand


def normalized_bbox(bbox, width, height):
    return [bbox[0] / width * 1000, bbox[1] / height * 1000,
            bbox[2] / width * 1000, bbox[3] / height * 1000]


def region_box(gt_bbox, mode="context", context_scale=1.3,
               min_area_ratio=0.10, max_area_ratio=0.25):
    if mode == "context":
        return expand(gt_bbox, context_scale)
    if mode != "adaptive":
        raise ValueError("unknown region target mode: " + mode)
    if not 0 < min_area_ratio <= max_area_ratio <= 1:
        raise ValueError("invalid adaptive area range")
    gt_w, gt_h = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    desired_area = min(max(gt_w * gt_h, min_area_ratio * 1000 * 1000),
                       max_area_ratio * 1000 * 1000)
    side = math.sqrt(desired_area)
    width, height = min(1000.0, max(gt_w, side)), min(1000.0, max(gt_h, side))
    cx, cy = (gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2
    x1 = max(0.0, min(1000.0 - width, cx - width / 2))
    y1 = max(0.0, min(1000.0 - height, cy - height / 2))
    return [x1, y1, x1 + width, y1 + height]


def region_target(gt_bbox, context_scale=1.3, mode="context",
                  min_area_ratio=0.10, max_area_ratio=0.25):
    box = region_box(gt_bbox, mode, context_scale, min_area_ratio, max_area_ratio)
    box = [math.floor(box[0]), math.floor(box[1]),
           math.ceil(box[2]), math.ceil(box[3])]
    return "<bbox>[%d,%d,%d,%d]</bbox>" % tuple(box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_recipe.yaml")
    parser.add_argument("--output", default="data/outputs/train_grpo_region.jsonl")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--target-mode", choices=["context", "adaptive"], default="context")
    parser.add_argument("--min-area-ratio", type=float, default=0.10)
    parser.add_argument("--max-area-ratio", type=float, default=0.25)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    entries = []
    for name, adapter in ADAPTERS.items():
        raw = os.path.join(cfg["corpus_dir"], name)
        if os.path.isdir(raw):
            entries.extend(adapter(raw))
    picked = sample_balanced(entries, cfg["total"] + cfg["val_size"],
                             cfg["seed"], cfg["per_image_cap"])
    train = picked[cfg["val_size"]:cfg["val_size"] + args.samples]
    if len(train) != args.samples:
        raise RuntimeError("requested %d samples, got %d" % (args.samples, len(train)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for entry in train:
            with Image.open(entry["image"]) as image:
                width, height = image.size
            gt_bbox = normalized_bbox(entry["bbox"], width, height)
            target_bbox = region_box(gt_bbox, args.target_mode,
                                     min_area_ratio=args.min_area_ratio,
                                     max_area_ratio=args.max_area_ratio)
            record = {"id": entry.get("id", entry["image"]),
                      "image": entry["image"], "instruction": entry["instruction"],
                      "mode": "region",
                      "target": region_target(gt_bbox, mode=args.target_mode,
                                              min_area_ratio=args.min_area_ratio,
                                              max_area_ratio=args.max_area_ratio),
                      "target_bbox": target_bbox, "target_mode": args.target_mode,
                      "gt_bbox": gt_bbox, "image_size": [width, height]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote %d region-GRPO records to %s" % (len(train), args.output))


if __name__ == "__main__":
    main()
