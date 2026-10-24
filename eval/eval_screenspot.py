# -*- coding: utf-8 -*-
"""Evaluate GUI grounding on ScreenSpot-v2 / ScreenSpot-Pro.

Hit criterion: predicted point inside the GT bbox (official ScreenSpot protocol).
Writes per-sample JSONL (fields: id, point, bbox, ui_type, platform, app, hit)
compatible with screenspot_pro_analysis/error_decomposition.py, plus a printed
summary split by ui_type and platform.

Examples:
  # ScreenSpot-Pro, ZonUI-3B weights, direct protocol (baseline reproduction)
  python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
      --model models/ZonUI-3B --mode direct --out outputs/zonui_pro.jsonl

  # ReAGround LoRA adapter, two-stage inference
  python eval/eval_screenspot.py --benchmark pro --data-dir benchmarks/screenspot_pro \
      --model models/Qwen2.5-VL-3B-Instruct --adapter outputs/group_d \
      --mode twostage --out outputs/group_d_pro.jsonl
"""
import argparse
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from infer.parse import hit


def iter_pro(data_dir):
    """Yields dicts with absolute-pixel bboxes from ScreenSpot-Pro annotations."""
    ann_dir = os.path.join(data_dir, "annotations")
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        for e in json.load(open(os.path.join(ann_dir, fname), encoding="utf-8")):
            yield {"id": e["id"],
                   "image_path": os.path.join(data_dir, "images", e["img_filename"]),
                   "bbox": [float(v) for v in e["bbox"]],
                   "instruction": e["instruction"],
                   "ui_type": e["ui_type"], "platform": e["platform"],
                   "app": e["application"]}


def iter_v2(data_dir):
    """Yields dicts from ScreenSpot-v2 parquets; bbox is normalized 0-1 there."""
    import pandas as pd
    from PIL import Image
    data = os.path.join(data_dir, "data")
    for fname in sorted(os.listdir(data)):
        if not fname.endswith(".parquet"):
            continue
        for _, r in pd.read_parquet(os.path.join(data, fname)).iterrows():
            img = Image.open(io.BytesIO(r["image"]["bytes"]))
            w, h = img.size
            x1, y1, x2, y2 = (float(v) for v in r["bbox"])
            yield {"id": r["file_name"],
                   "image_path": None, "image_bytes": r["image"]["bytes"],
                   "bbox": [x1 * w, y1 * h, x2 * w, y2 * h],
                   "instruction": r["instruction"],
                   "ui_type": r["data_type"], "platform": r["data_source"],
                   "app": r["data_source"]}


def iter_custom(path):
    """Held-out validation JSONL produced by data/build_dataset.py."""
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        yield {"id": e["id"], "image_path": e["image"], "bbox": e["bbox"],
               "instruction": e["instruction"], "ui_type": e.get("ui_type", ""),
               "platform": e.get("platform", ""), "app": e.get("platform", "")}


ITERATORS = {"pro": iter_pro, "v2": iter_v2}


def summarize(rows):
    def acc(rows):
        n = len(rows)
        return "%d/%d (%.1f%%)" % (sum(r["hit"] for r in rows), n,
                                   100.0 * sum(r["hit"] for r in rows) / n) if n else "0/0"
    print("\n==== summary ====")
    print("overall:", acc(rows))
    for key in ("ui_type", "platform", "app"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        print("-- by %s --" % key)
        for k in sorted(groups):
            print("  %-22s %s" % (k, acc(groups[k])))
    n_fail = sum(1 for r in rows if r["point"] is None)
    print("parse failures (counted as miss): %d" % n_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["pro", "v2", "custom"])
    ap.add_argument("--data-dir", help="benchmark root (pro/v2)")
    ap.add_argument("--custom-jsonl", help="held-out set from build_dataset.py")
    ap.add_argument("--model", help="base model path (required unless --dry-run)")
    ap.add_argument("--adapter", help="LoRA adapter path (optional)")
    ap.add_argument("--mode", default="direct", choices=["direct", "twostage", "grpo"])
    ap.add_argument("--out", help="per-sample JSONL output path")
    ap.add_argument("--limit", type=int, help="evaluate only the first N samples")
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation, e.g. sdpa or flash_attention_2")
    ap.add_argument("--dry-run", action="store_true",
                    help="iterate data only: check counts and image paths, no model")
    args = ap.parse_args()

    if args.benchmark == "custom" and not args.custom_jsonl:
        ap.error("--benchmark custom requires --custom-jsonl")
    if args.benchmark == "custom":
        samples = list(iter_custom(args.custom_jsonl))
    else:
        samples = list(ITERATORS[args.benchmark](args.data_dir))
    if args.limit:
        samples = samples[:args.limit]

    if args.dry_run:
        from PIL import Image
        print("samples:", len(samples))
        for s in samples[:3]:
            img = Image.open(s["image_path"]) if s.get("image_path") \
                else Image.open(io.BytesIO(s["image_bytes"]))
            print({"id": s["id"], "size": img.size, "bbox": s["bbox"],
                   "ui_type": s["ui_type"], "platform": s["platform"],
                   "instruction": s["instruction"][:60]})
        missing = sum(1 for s in samples
                      if s.get("image_path") and not os.path.exists(s["image_path"]))
        print("missing images:", missing)
        return

    from protocol import load_model
    from infer.pipeline import ground_direct, ground_grpo, ground_twostage
    from PIL import Image

    model, processor = load_model(args.model, args.adapter, args.attn)
    ground = {"direct": ground_direct, "twostage": ground_twostage,
              "grpo": ground_grpo}[args.mode]

    rows = []
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    for i, s in enumerate(samples):
        image = Image.open(s["image_path"]).convert("RGB") if s.get("image_path") \
            else Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
        point, info = ground(model, processor, image, s["instruction"])
        row = {"id": s["id"], "point": point, "bbox": s["bbox"],
               "ui_type": s["ui_type"], "platform": s["platform"], "app": s["app"],
               "hit": hit(point, s["bbox"]), "info": info}
        rows.append(row)
        if out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
        if (i + 1) % 50 == 0:
            print("%d/%d done, running acc %.1f%%"
                  % (i + 1, len(samples),
                     100.0 * sum(r["hit"] for r in rows) / len(rows)))
    if out_f:
        out_f.close()
    summarize(rows)


if __name__ == "__main__":
    main()
