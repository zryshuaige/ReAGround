# -*- coding: utf-8 -*-
"""Unit tests for ReAGround. No GPU needed.

  python tests/test_core.py                  # pure-python + math tests
  REAGROUND_MODEL=/path/to/Qwen2.5-VL-3B-Instruct python tests/test_core.py
      # additionally: tokenize_target with the real tokenizer, sequence builders
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("USE_TF", "0")  # skip transformers' TF/Keras probe

import torch
import torch.nn.functional as F

import infer.parse


def test_parse():
    p = infer.parse
    assert p.parse_point("<point>(1842, 47)</point>") == (1842.0, 47.0)
    assert p.parse_point("[300, 235]") == (300.0, 235.0)
    assert p.parse_point("[1, 2, 3, 4]") is None
    assert p.parse_point("no point") is None
    assert p.parse_area("<area>[0.31, 0.00, 0.58, 0.09]</area>") == [0.31, 0.0, 0.58, 0.09]
    assert p.parse_area("<area>[1.31, 0, 0.58, 0.09]</area>") is None
    crop = p.area_to_crop([0.25, 0.0, 0.5, 0.5], 2560, 1664)
    x, y = p.point_from_original(700, 300, (896, 1232), crop)
    ox, oy = p.point_to_original(x, y, (896, 1232), crop)
    assert abs(ox - 700) < 1e-6 and abs(oy - 300) < 1e-6
    assert p.hit((10, 10), [5, 5, 20, 20]) and not p.hit((30, 10), [5, 5, 20, 20])
    print("parse: OK")


def test_compute_loss():
    from train.train_lora import SectionalTrainer
    V, L = 50, 12
    logits = torch.full((1, L, V), -10.0)
    labels = torch.full((1, L), -100)
    labels[0, 2:] = torch.arange(2, 12) % V
    for t in range(L - 1):
        if labels[0, t + 1] != -100:
            logits[0, t, labels[0, t + 1]] = 5.0
    sections = torch.zeros((1, L), dtype=torch.long)
    sections[0, 3:6], sections[0, 6:9], sections[0, 9:11], sections[0, 11] = 1, 2, 3, 4

    class FakeModel(torch.nn.Module):
        def forward(self, **kw):
            return types.SimpleNamespace(logits=logits)

    trainer = SectionalTrainer.__new__(SectionalTrainer)
    trainer.section_weights = {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}
    loss = trainer.compute_loss(FakeModel(), {
        "input_ids": torch.zeros((1, L), dtype=torch.long),
        "labels": labels, "loss_sections": sections,
        "attention_mask": torch.ones((1, L))})
    ce = F.cross_entropy(logits[0, :-1].float(), labels[0, 1:].clamp_min(0),
                         reduction="none") * (labels[0, 1:] != -100)
    sec = sections[0, 1:]
    exp = sum(w * ce[sec == sid].sum() / (sec == sid).sum()
              for sid, w in [(1, 0.0), (2, 1.0), (3, 1.0), (4, 1.0)])
    assert abs(loss.item() - float(exp)) < 1e-5
    print("compute_loss: OK (matches hand-computed section-weighted mean)")


def test_tokenize_and_builders(model_path, sample_image=None):
    from transformers import AutoProcessor
    from train.dataset import tokenize_target, _spans
    tok = AutoProcessor.from_pretrained(model_path).tokenizer
    for target in ['<think>abc</think><region>top toolbar</region>'
                   '<area>[0.31, 0.00, 0.58, 0.09]</area>',
                   '<point>(1842, 47)</point>', '(1842, 47)']:
        ids, secs = tokenize_target(tok, target)
        assert tok.decode(ids) == target + "<|im_end|>\n"
        assert sum(e - s for s, e, _ in _spans(target + "<|im_end|>\n")) == \
            len(target + "<|im_end|>\n")
    print("tokenize_target: OK (real tokenizer, exact decode round-trip)")

    if sample_image:
        import json
        from PIL import Image
        from data.build_dataset import build_sequences
        from protocol import resized_dims
        from infer.parse import parse_point, point_to_original
        entry = {"image": sample_image, "instruction": "Refresh the file explorer.",
                 "bbox": [473, 183, 503, 219], "ui_type": "icon", "platform": "desktop"}
        r2 = build_sequences(entry, "s00000")[1]
        px, py = parse_point(r2["target"])
        crop = r2["crop"]
        ox, oy = point_to_original(px, py, resized_dims(crop[2] - crop[0],
                                                        crop[3] - crop[1]), crop)
        assert abs(ox - 488) < 2 and abs(oy - 201) < 2, (ox, oy)
        print("build_sequences: OK (round-2 target maps back to GT center)")


if __name__ == "__main__":
    test_parse()
    test_compute_loss()
    model = os.environ.get("REAGROUND_MODEL")
    if model:
        test_tokenize_and_builders(model, os.environ.get("REAGROUND_TEST_IMAGE"))
    else:
        print("skip tokenizer/builder tests (set REAGROUND_MODEL to enable)")
    print("all tests passed")
