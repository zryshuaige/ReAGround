# -*- coding: utf-8 -*-
"""Geometry rewards for normalized inspection boxes; no GPU dependencies."""
import math
import re

BOX_RE = re.compile(
    r"\s*<bbox>\s*\[\s*([^]]+?)\s*\]\s*</bbox>\s*", re.DOTALL)


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[-1].get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def parse_bbox(text):
    match = BOX_RE.fullmatch(completion_text(text))
    if not match:
        return None
    try:
        box = [float(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        return None
    x1, y1, x2, y2 = box
    return box if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000 else None


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def expand(box, scale):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw = (box[2] - box[0]) * scale / 2
    hh = (box[3] - box[1]) * scale / 2
    return [max(0.0, cx - hw), max(0.0, cy - hh),
            min(1000.0, cx + hw), min(1000.0, cy + hh)]


def reward_metrics(completion, gt_bbox, max_area_ratio=0.25, context_scale=1.3):
    box = parse_bbox(completion)
    if box is None:
        return 0.0, 0.0, 0.0, 0.0
    context = expand(gt_bbox, context_scale)
    ix1, iy1 = max(box[0], context[0]), max(box[1], context[1])
    ix2, iy2 = min(box[2], context[2]), min(box[3], context[3])
    intersection = area([ix1, iy1, ix2, iy2])
    coverage = min(1.0, intersection / area(context)) if area(context) else 0.0
    max_area = max(area(context), max_area_ratio * 1000 * 1000)
    success = coverage >= 0.999 and area(box) <= max_area
    efficiency = 0.0
    if success:
        efficiency = 1.0 if max_area <= area(context) else min(
            1.0, max(0.0, (max_area - area(box)) / (max_area - area(context))))
    return 0.1, 0.4 * coverage, 0.5 * float(success), 0.5 * efficiency


def _reward(completions, gt_bbox, index, max_area_ratio=0.25,
            context_scale=1.3, **kwargs):
    return [reward_metrics(c, gt, max_area_ratio, context_scale)[index]
            for c, gt in zip(completions, gt_bbox)]


def format_reward(completions, gt_bbox, **kwargs):
    return _reward(completions, gt_bbox, 0, **kwargs)


def coverage_reward(completions, gt_bbox, **kwargs):
    return _reward(completions, gt_bbox, 1, **kwargs)


def containment_reward(completions, gt_bbox, **kwargs):
    return _reward(completions, gt_bbox, 2, **kwargs)


def size_reward(completions, gt_bbox, **kwargs):
    return _reward(completions, gt_bbox, 3, **kwargs)


REWARD_FUNCS = [format_reward, coverage_reward, containment_reward, size_reward]
