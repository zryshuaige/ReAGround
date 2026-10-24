# -*- coding: utf-8 -*-
"""Inference pipelines: direct (baseline) and two-stage region-anchored (ReAGround)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from protocol import generate, PATCH_FACTOR
from infer.parse import parse_point, parse_area, area_to_crop, point_to_original
from train.grpo_reward import parse_bbox


def ground_direct(model, processor, image, instruction):
    """Single-round grounding, ZonUI-3B protocol. Returns ((x, y) | None, info)."""
    out, (rw, rh) = generate(model, processor, image, "direct", instruction)
    pt = parse_point(out)
    if pt is None:
        return None, {"raw": out}
    return (pt[0] / rw * image.width, pt[1] / rh * image.height), {"raw": out}


def ground_twostage(model, processor, image, instruction):
    """ReAGround: round 1 anchors an <area>, round 2 points inside the crop.

    The crop is *not* manually upscaled: the processor's smart_resize with
    MIN_PIXELS automatically enlarges small crops -- that is the zoom-in.
    Returns ((x, y) in original pixels | None, info).
    """
    out1, _ = generate(model, processor, image, "round1", instruction,
                       max_new_tokens=512)
    area = parse_area(out1)
    if area is None:
        area = [0.0, 0.0, 1.0, 1.0]  # fall back to the full image
    crop_box = area_to_crop(area, image.width, image.height)
    if (crop_box[2] - crop_box[0] < PATCH_FACTOR
            or crop_box[3] - crop_box[1] < PATCH_FACTOR):
        crop_box = (0, 0, image.width, image.height)  # unusable sliver: degrade to direct
    crop = image.crop(crop_box)
    out2, resized_wh = generate(model, processor, crop, "round2", instruction)
    pt = parse_point(out2)
    info = {"round1": out1, "round2": out2, "area": area}
    if pt is None:
        return None, info
    return point_to_original(pt[0], pt[1], resized_wh, crop_box), info


def ground_grpo(model, processor, image, instruction):
    """A-initialized inspection box followed by A-style direct pointing."""
    raw, _ = generate(model, processor, image, "region", instruction)
    box = parse_bbox(raw)
    info = {"region_raw": raw, "region_bbox": box}
    if box is None:
        return None, info
    crop_box = (round(box[0] / 1000 * image.width),
                round(box[1] / 1000 * image.height),
                round(box[2] / 1000 * image.width),
                round(box[3] / 1000 * image.height))
    if crop_box[2] - crop_box[0] < 1 or crop_box[3] - crop_box[1] < 1:
        return None, info
    crop = image.crop(crop_box)
    point, click_info = ground_direct(model, processor, crop, instruction)
    info.update({"crop_box": crop_box, "click_raw": click_info["raw"]})
    if point is None:
        return None, info
    return (crop_box[0] + point[0], crop_box[1] + point[1]), info
