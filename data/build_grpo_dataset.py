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


def region_target(gt_bbox, context_scale=1.3):
    context = expand(gt_bbox, context_scale)
    box = [math.floor(context[0]), math.floor(context[1]),
           math.ceil(context[2]), math.ceil(context[3])]
    return "<bbox>[%d,%d,%d,%d]</bbox>" % tuple(box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_recipe.yaml")
    parser.add_argument("--output", default="data/outputs/train_grpo_region.jsonl")
    parser.add_argument("--samples", type=int, default=2000)
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
            record = {"id": entry.get("id", entry["image"]),
                      "image": entry["image"], "instruction": entry["instruction"],
                      "mode": "region", "target": region_target(gt_bbox),
                      "gt_bbox": gt_bbox, "image_size": [width, height]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote %d region-GRPO records to %s" % (len(train), args.output))


if __name__ == "__main__":
    main()
