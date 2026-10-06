# -*- coding: utf-8 -*-
"""Build ReAGround SFT data from the four raw corpora.

Pipeline: raw corpora -> normalized entries -> balanced 8K sample (fixed seed,
per-image cap) -> two-stage supervision sequences (train.jsonl) + held-out
val.jsonl for lambda selection.

Normalized entry: {"image": repo-relative path, "instruction": str,
                   "bbox": [x1,y1,x2,y2] absolute px, "ui_type": "icon"|"text"|"",
                   "platform": "web"|"mobile"|"desktop"}

Coordinate scales differ per corpus (ShowUI: [0,1]; UGround: [0,999]) and are
converted to absolute pixels here -- this is the single place where it happens.

  python data/build_dataset.py --config configs/data_recipe.yaml
  python data/build_dataset.py --config configs/data_recipe.yaml --show-sample 3
"""
import argparse
import glob
import io
import json
import os
import random
import re

import pandas as pd
import yaml
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from protocol import PROMPTS, resized_dims

IMG_OUT = os.path.join("corpus", "processed", "images")


# ---------------- raw corpus adapters (schema per source, see README) ----------------

def _materialize(image_field, corpus, key):
    """Write embedded image bytes to disk once; return repo-relative path."""
    path = os.path.join(IMG_OUT, corpus, key + ".png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = image_field["bytes"] if isinstance(image_field, dict) else image_field
        Image.open(io.BytesIO(data)).convert("RGB").save(path)
    return path


def iter_showui_desktop(raw):
    """showlab/ShowUI-desktop: one row per element; bbox/point normalized [0,1]."""
    for f in sorted(glob.glob(os.path.join(raw, "**", "*.parquet"), recursive=True)):
        for i, r in pd.read_parquet(f).iterrows():
            path = _materialize(r["image"], "showui_desktop", "desktop_%s_%s" % (os.path.basename(f), i))
            w, h = Image.open(path).size
            x1, y1, x2, y2 = (float(v) for v in r["bbox"])
            yield {"image": path, "instruction": str(r["instruction"]),
                   "bbox": [x1 * w, y1 * h, x2 * w, y2 * h],
                   "ui_type": "", "platform": "desktop"}


def iter_showui_web(raw):
    """showlab/ShowUI-web: one row per screenshot, parallel lists per element;
    bbox normalized [0,1]; images extracted from image.tar.gz by image_url.

    The corpus ships element *descriptions* (visible text), not task-style
    queries, so we synthesize the query with a template -- documented choice.
    """
    for f in sorted(glob.glob(os.path.join(raw, "**", "*.parquet"), recursive=True)):
        for _, r in pd.read_parquet(f).iterrows():
            path = os.path.join(raw, r["image_url"])
            if not os.path.exists(path):
                continue  # image.tar.gz not extracted yet
            w, h = Image.open(path).size
            for text, bbox, etype in zip(r["instruction"], r["bbox"], r["type"]):
                text = str(text).strip()
                if text:
                    q = 'Click the %s "%s"' % (str(etype).lower(), text)
                    ui_type = "text"
                else:
                    q = "Click the %s element" % str(etype).lower()
                    ui_type = "icon"
                x1, y1, x2, y2 = (float(v) for v in bbox)
                yield {"image": path, "instruction": q,
                       "bbox": [x1 * w, y1 * h, x2 * w, y2 * h],
                       "ui_type": ui_type, "platform": "web"}


def iter_uground(raw):
    """osunlp/UGround-V1-Data-Box (Web-Hybrid, bbox version; GATED).
    Coordinates normalized to [0,999]. The schema must be verified with
    --show-sample on first download: unexpected columns raise instead of
    silently degrading the instructions."""
    for f in sorted(glob.glob(os.path.join(raw, "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(f)
        if not {"bbox", "width", "height", "image"} <= set(df.columns):
            raise KeyError("unexpected UGround schema in %s: %s -- adapt iter_uground"
                           % (os.path.basename(f), list(df.columns)))
        for i, r in df.iterrows():
            instr = None
            conv = r.get("conversations")
            if isinstance(conv, str):
                conv = json.loads(conv)
            for turn in conv or []:
                if turn.get("from") == "human":
                    instr = re.sub(r"<image>|<img>.*?</img>", "", turn.get("value", "")).strip()
            instr = instr or r.get("instruction") or r.get("query")
            if not instr:
                raise KeyError("no instruction column in %s: %s -- adapt iter_uground"
                               % (os.path.basename(f), list(df.columns)))
            vals = [float(v) for v in r["bbox"][:4]]
            x1, y1, x2, y2 = [v / 999.0 for v in vals]
            w, h = int(r["width"]), int(r["height"])
            path = _materialize(r["image"], "uground", "ug_%s_%s" % (os.path.basename(f), i))
            yield {"image": path, "instruction": instr,
                   "bbox": [x1 * w, y1 * h, x2 * w, y2 * h],
                   "ui_type": "", "platform": "web"}


def iter_amex(raw):
    """AMEX prepared via ShowUI's prepare/hf_amex.py: JSONL with image path,
    element bbox (absolute px) and functionality description."""
    for f in sorted(glob.glob(os.path.join(raw, "**", "*.json*"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            img = r.get("image") or r.get("img_path") or r.get("img")
            bbox = r.get("bbox") or r.get("box")
            instr = r.get("instruction") or r.get("functionality") or r.get("description")
            if not (img and bbox and instr):
                raise KeyError("unexpected AMEX record, adapt iter_amex: %s" % list(r))
            path = img if os.path.isabs(img) else os.path.join(raw, img)
            yield {"image": path, "instruction": str(instr),
                   "bbox": [float(v) for v in bbox[:4]],
                   "ui_type": "", "platform": "mobile"}


ADAPTERS = {"showui_web": iter_showui_web, "showui_desktop": iter_showui_desktop,
            "uground": iter_uground, "amex": iter_amex}


# ---------------- two-stage label construction (all rule-based) ----------------

def region_name(cx, cy):
    """Functional area name from the normalized GT center: layout prior."""
    if cy < 0.12:
        return "top toolbar / menu bar"
    if cy > 0.93:
        return "bottom status bar"
    if cx < 0.20:
        return "left sidebar / panel"
    if cx > 0.80:
        return "right sidebar / panel"
    return "main content area"


def area_box(bbox, width, height, cols=4, rows=2):
    """Grid cell (cols x rows) containing the GT center, unioned with the GT
    box, clipped to the image; normalized [0,1]. Deliberately coarse."""
    cx = (bbox[0] + bbox[2]) / 2 / width
    cy = (bbox[1] + bbox[3]) / 2 / height
    gx = min(int(cx * cols), cols - 1)
    gy = min(int(cy * rows), rows - 1)
    x1, x2 = gx / cols, (gx + 1) / cols
    y1, y2 = gy / rows, (gy + 1) / rows
    x1 = min(x1, bbox[0] / width)
    y1 = min(y1, bbox[1] / height)
    x2 = max(x2, bbox[2] / width)
    y2 = max(y2, bbox[3] / height)
    return [round(max(0.0, x1), 2), round(max(0.0, y1), 2),
            round(min(1.0, x2), 2), round(min(1.0, y2), 2)]


def build_sequences(entry, sid):
    """One entry -> two SFT records (round 1: anchor, round 2: in-region point)."""
    w, h = Image.open(entry["image"]).size
    bbox = entry["bbox"]
    gcx, gcy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    region = region_name(gcx / w, gcy / h)
    area = area_box(bbox, w, h)
    crop = (round(area[0] * w), round(area[1] * h), round(area[2] * w), round(area[3] * h))
    rw, rh = resized_dims(crop[2] - crop[0], crop[3] - crop[1])
    px = (gcx - crop[0]) / (crop[2] - crop[0]) * rw
    py = (gcy - crop[1]) / (crop[3] - crop[1]) * rh
    think = "The target of \"%s\" is typically located in the %s." % (
        entry["instruction"], region)
    r1 = {"id": sid + "-r1", "mode": "round1", "image": entry["image"],
          "instruction": entry["instruction"],
          "target": "<think>%s</think><region>%s</region><area>%s</area>"
                    % (think, region, json.dumps(area))}
    r2 = {"id": sid + "-r2", "mode": "round2", "image": entry["image"],
          "crop": list(crop), "instruction": entry["instruction"],
          "target": "<point>(%d, %d)</point>" % (round(px), round(py))}
    return [r1, r2]


def build_direct(entry, sid):
    """Group-A baseline format: single record, bare point like ZonUI training."""
    w, h = Image.open(entry["image"]).size
    bbox = entry["bbox"]
    rw, rh = resized_dims(w, h)
    px = (bbox[0] + bbox[2]) / 2 / w * rw
    py = (bbox[1] + bbox[3]) / 2 / h * rh
    return [{"id": sid + "-d", "mode": "direct", "image": entry["image"],
             "instruction": entry["instruction"],
             "target": "(%d, %d)" % (round(px), round(py))}]


# ---------------- sampling ----------------

def sample_balanced(entries, total, seed, per_image_cap):
    """Platform-balanced random sample (web/mobile/desktop ~= 1:1:1, ZonUI-style),
    no manual filtering; cap how many entries one screenshot may contribute."""
    rng = random.Random(seed)
    by_platform = {}
    for e in entries:
        by_platform.setdefault(e["platform"], []).append(e)
    quota = total // len(by_platform)
    picked = []
    for platform, pool in sorted(by_platform.items()):
        rng.shuffle(pool)
        count_per_image = {}
        n = 0
        for e in pool:
            if n >= quota:
                break
            if count_per_image.get(e["image"], 0) >= per_image_cap:
                continue
            count_per_image[e["image"]] = count_per_image.get(e["image"], 0) + 1
            picked.append(e)
            n += 1
        print("  platform %-8s quota %d, picked %d" % (platform, quota, n))
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_recipe.yaml")
    ap.add_argument("--fmt", default="twostage", choices=["twostage", "direct"],
                    help="twostage = ReAGround groups; direct = group-A baseline")
    ap.add_argument("--show-sample", type=int, metavar="N",
                    help="print N normalized entries per corpus and exit")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    entries = []
    for name, adapter in ADAPTERS.items():
        raw = os.path.join(cfg["corpus_dir"], name)
        if not os.path.isdir(raw):
            print("skip %s (not downloaded)" % name)
            continue
        sub = list(adapter(raw))
        print("%-16s %8d entries" % (name, len(sub)))
        if args.show_sample:
            for e in sub[:args.show_sample]:
                print("  ", json.dumps({k: (v if k != "instruction" else v[:60])
                                        for k, v in e.items()}, ensure_ascii=False))
        entries.extend(sub)
    if args.show_sample:
        return

    picked = sample_balanced(entries, cfg["total"] + cfg["val_size"],
                             cfg["seed"], cfg["per_image_cap"])
    val, train = picked[:cfg["val_size"]], picked[cfg["val_size"]:]
    print("sampled: %d train + %d val" % (len(train), len(val)))

    os.makedirs("data/outputs", exist_ok=True)
    builder = build_sequences if args.fmt == "twostage" else build_direct
    n_rec = 0
    with open("data/outputs/train_%s.jsonl" % args.fmt, "w", encoding="utf-8") as f:
        for i, e in enumerate(train):
            for rec in builder(e, "s%05d" % i):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
    with open("data/outputs/val.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(val):
            f.write(json.dumps({"id": "v%05d" % i, "image": e["image"],
                                "bbox": e["bbox"], "instruction": e["instruction"],
                                "ui_type": e["ui_type"], "platform": e["platform"]},
                               ensure_ascii=False) + "\n")
    print("wrote data/outputs/train_%s.jsonl (%d sequences) and val.jsonl (%d)"
          % (args.fmt, n_rec, len(val)))


if __name__ == "__main__":
    main()
