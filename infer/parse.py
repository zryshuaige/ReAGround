# -*- coding: utf-8 -*-
"""Parsing and coordinate mapping for ReAGround.

Frame conventions (Qwen2.5-VL style):
  - <area>  : normalized [x1, y1, x2, y2] in [0, 1] w.r.t. the *original* image
  - <point> : pixels in the *resized* frame of the image the model received
Mapping chain at inference: resized-crop pixels -> crop-normalized -> original pixels.

Run `python infer/parse.py` for self-tests.
"""
import ast
import re


def parse_point(text):
    """(x, y) floats from '<point>(x, y)</point>', bare '(x, y)' or '[x, y]';
    None on failure. Bracket groups with != 2 numbers (e.g. an <area> box)
    are skipped in favor of the first true point."""
    for m in re.finditer(r"[\(\[]\s*([\d.,\s]+?)\s*[\)\]]", text):
        try:
            vals = ast.literal_eval("[" + m.group(1) + "]")
        except (ValueError, SyntaxError):
            continue
        if len(vals) == 2:
            return float(vals[0]), float(vals[1])
    return None


def parse_area(text):
    """Normalized [x1, y1, x2, y2] from '<area>[...]</area>'; None on failure."""
    m = re.search(r"<area>\s*\[([\d.,\s]+)\]", text)
    if not m:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in ast.literal_eval("[" + m.group(1) + "]"))
    except (ValueError, SyntaxError):
        return None
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return [x1, y1, x2, y2]


def area_to_crop(area, width, height):
    """Normalized area -> integer pixel crop box (x0, y0, x1, y1)."""
    return (round(area[0] * width), round(area[1] * height),
            round(area[2] * width), round(area[3] * height))


def point_to_original(px, py, resized_wh, crop_box):
    """Point in resized-crop pixels -> original-image pixels."""
    rw, rh = resized_wh
    x0, y0, x1, y1 = crop_box
    return x0 + px / rw * (x1 - x0), y0 + py / rh * (y1 - y0)


def point_from_original(px, py, resized_wh, crop_box):
    """Inverse of point_to_original (used to build round-2 training targets)."""
    rw, rh = resized_wh
    x0, y0, x1, y1 = crop_box
    return (px - x0) / (x1 - x0) * rw, (py - y0) / (y1 - y0) * rh


def hit(point, bbox):
    """Official ScreenSpot criterion: predicted point inside the GT bbox."""
    x1, y1, x2, y2 = bbox
    return point is not None and x1 <= point[0] <= x2 and y1 <= point[1] <= y2


if __name__ == "__main__":
    assert parse_point("<point>(1842, 47)</point>") == (1842.0, 47.0)
    assert parse_point("(488.0, 201.0)") == (488.0, 201.0)
    assert parse_point("<point>(488.5, 201)</point>") == (488.5, 201.0)
    assert parse_point("[300, 235]") == (300.0, 235.0)   # base model's bracket habit
    assert parse_point("[1, 2, 3, 4]") is None           # a box is not a point
    assert parse_point("<area>[0.1, 0.2, 0.3, 0.4]</area> then (50, 60)") == (50.0, 60.0)
    assert parse_point("no point here") is None
    assert parse_point("(1, 2, 3)") is None
    assert parse_area("<area>[0.31, 0.00, 0.58, 0.09]</area>") == [0.31, 0.0, 0.58, 0.09]
    assert parse_area("<area>[1.31, 0.00, 0.58, 0.09]</area>") is None
    assert parse_area("<area>[0.5, 0.5, 0.4, 0.9]</area>") is None
    assert parse_area("garbage") is None
    # round trip: original -> crop-resized -> original
    crop = area_to_crop([0.25, 0.0, 0.5, 0.5], 2560, 1664)
    rw, rh = 896, 1232  # pretend resized crop dims
    x, y = point_from_original(700, 300, (rw, rh), crop)
    ox, oy = point_to_original(x, y, (rw, rh), crop)
    assert abs(ox - 700) < 1e-6 and abs(oy - 300) < 1e-6
    assert hit((10, 10), [5, 5, 20, 20]) and not hit((30, 10), [5, 5, 20, 20])
    assert not hit(None, [5, 5, 20, 20])
    print("parse.py: all self-tests passed")
