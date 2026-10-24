# -*- coding: utf-8 -*-
"""Build region-GRPO records from the exact run-1 sampling recipe."""
import argparse
import json
import os
import sys

import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import ADAPTERS, sample_balanced


def normalized_bbox(bbox, width, height):
    return [bbox[0] / width * 1000, bbox[1] / height * 1000,
            bbox[2] / width * 1000, bbox[3] / height * 1000]


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
            record = {"id": entry.get("id", entry["image"]),
                      "image": entry["image"], "instruction": entry["instruction"],
                      "gt_bbox": normalized_bbox(entry["bbox"], width, height),
                      "image_size": [width, height]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote %d region-GRPO records to %s" % (len(train), args.output))


if __name__ == "__main__":
    main()
