# -*- coding: utf-8 -*-
"""CPU tests for region rewards and the two-call GRPO inference pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.build_grpo_dataset import normalized_bbox, region_box, region_target
from infer.pipeline import ground_dual, ground_grpo
from protocol import MAX_PIXELS
from train.grpo_reward import ADAPTIVE_REWARD_FUNCS, REWARD_FUNCS, area, parse_bbox
from train.grpo_reward import adaptive_reward_metrics, reward_metrics
from train.train_grpo import validate_dry_run, validate_prompt_budget, vllm_engine_kwargs


def total(completion, gt):
    return sum(reward_metrics(completion, gt))


def test_reward():
    gt = [400, 400, 600, 600]
    context = [370, 370, 630, 630]
    assert parse_bbox("<bbox>[400,400,600,600]</bbox>") == [400, 400, 600, 600]
    assert parse_bbox("<bbox>[0,0,1001,600]</bbox>") is None
    assert parse_bbox("<bbox>[0,0,500,500]</bbox> extra") is None
    assert parse_bbox("not a box") is None
    assert total("<bbox>[370,370,630,630]</bbox>", gt) == 1.5
    assert total("<bbox>[0,0,1000,1000]</bbox>", gt) == 0.5
    assert total("<bbox>[400,400,600,500]</bbox>", gt) < 0.5
    assert total("<bbox>[300,300,700,700]</bbox>", gt) > 1.0
    assert total("garbage", gt) == 0.0
    conversational = [{"role": "assistant", "content": "<bbox>[370,370,630,630]</bbox>"}]
    assert total(conversational, gt) == 1.5
    rewards = [func([conversational], [gt])[0] for func in REWARD_FUNCS]
    assert rewards == [0.1, 0.4, 0.5, 0.5]
    assert context == [370.0, 370.0, 630.0, 630.0]
    print("reward: OK")


def test_adaptive_reward():
    gt = [790, 790, 810, 810]
    target_box = region_box(gt, mode="adaptive")
    assert abs(area(target_box) - 100000) < 1e-6
    assert target_box[0] <= gt[0] and target_box[1] <= gt[1]
    assert target_box[2] >= gt[2] and target_box[3] >= gt[3]
    target = region_target(gt, mode="adaptive")
    metrics = adaptive_reward_metrics(target, gt, target_box)
    assert metrics[0] == 0.1 and metrics[1] > 0.39
    assert metrics[2] == 0.5 and metrics[3] > 0.49
    far = adaptive_reward_metrics("<bbox>[0,0,400,300]</bbox>", gt, target_box)
    assert far[2] == 0.0 and far[3] == 0.0 and sum(far) < 0.5
    full = adaptive_reward_metrics("<bbox>[0,0,1000,1000]</bbox>", gt, target_box)
    assert full[3] == 0.0 and sum(full) < sum(metrics)
    rewards = [func([target], [gt], [target_box])[0] for func in ADAPTIVE_REWARD_FUNCS]
    assert all(abs(a - b) < 1e-9 for a, b in zip(rewards, metrics))
    boundary = region_box([0, 0, 20, 20], mode="adaptive")
    assert boundary[:2] == [0.0, 0.0] and abs(area(boundary) - 100000) < 1e-6
    bottom_gt = [535.6918633632461, 972.1426482153912, 711.4546693540387, 987.5255178738543]
    bottom_box = region_box(bottom_gt, mode="adaptive")
    bottom_metrics = adaptive_reward_metrics(
        region_target(bottom_gt, mode="adaptive"), bottom_gt, bottom_box)
    assert bottom_metrics[2] == 0.5 and bottom_metrics[3] > 0.49
    print("adaptive target and gated area reward: OK")


def test_normalized_bbox():
    assert normalized_bbox([50, 25, 100, 75], 200, 100) == [250, 250, 500, 750]
    target = region_target([400, 400, 600, 600])
    assert target == "<bbox>[370,370,630,630]</bbox>"
    assert total(target, [400, 400, 600, 600]) == 1.5
    print("normalized bbox and region SFT target: OK")


def test_vllm_engine_kwargs():
    original = {"max_model_len": None, "limit_mm_per_prompt": {"image": 1},
                "mm_processor_kwargs": {"min_pixels": 100}}
    patched = vllm_engine_kwargs(original, {"max_model_len": 4096})
    assert patched["max_model_len"] == 4096
    assert patched["limit_mm_per_prompt"] == {"image": 1, "video": 0}
    assert patched["mm_processor_kwargs"] == {"min_pixels": 100, "max_pixels": MAX_PIXELS}
    assert original["max_model_len"] is None
    assert vllm_engine_kwargs({"max_model_len": 8192}, {"max_model_len": 4096})["max_model_len"] == 8192
    print("vLLM engine arguments: OK")


def test_dry_run_gate():
    good = [{"grad_norm": 0.2, "reward": 0.4, "frac_reward_zero_std": 0.5},
            {"grad_norm": 0.1, "reward": 0.3, "frac_reward_zero_std": 0.25}]
    validate_dry_run(good)
    bad = [{"grad_norm": 0.0, "reward": 0.0, "frac_reward_zero_std": 1.0}] * 2
    try:
        validate_dry_run(bad)
    except RuntimeError as error:
        assert "no learning signal" in str(error)
    else:
        raise AssertionError("zero-signal dry-run must fail")
    validate_prompt_budget((4032, "boundary"), 4096)
    try:
        validate_prompt_budget((4033, "too-long"), 4096)
    except ValueError as error:
        assert "too-long" in str(error)
    else:
        raise AssertionError("over-length prompt must fail")
    print("dry-run learning-signal and prompt-budget gates: OK")


def test_pipeline(monkeypatch=None):
    from PIL import Image
    import infer.pipeline as pipeline
    calls = []
    bad_point = False

    def fake_generate(model, processor, image, mode, instruction, max_new_tokens=64):
        calls.append((mode, image.size))
        if mode == "region":
            return "<bbox>[100,0,500,500]</bbox>", image.size
        if bad_point:
            return "(%d,%d)" % (image.width * 2, image.height * 2), image.size
        return "(%d,%d)" % (image.width // 2, image.height // 2), image.size

    class DualModel:
        def __init__(self):
            self.active = []

        def set_adapter(self, name):
            self.active.append(name)

    original = pipeline.generate
    pipeline.generate = fake_generate
    try:
        point, info = ground_grpo(None, None, Image.new("RGB", (1000, 500)), "open Python")
        dual_model = DualModel()
        dual_point, _ = ground_dual(dual_model, None, Image.new("RGB", (1000, 500)), "open Python")
        bad_point = True
        legacy_bad, _ = ground_grpo(None, None, Image.new("RGB", (1000, 500)), "open Python")
        invalid, invalid_info = ground_dual(DualModel(), None, Image.new("RGB", (1000, 500)), "open Python")
    finally:
        pipeline.generate = original
    assert point == (300.0, 125.0)
    assert calls[:2] == [("region", (1000, 500)), ("direct", (400, 250))]
    assert info["crop_box"] == (100, 0, 500, 250)
    assert dual_point == point and dual_model.active == ["region", "direct"]
    assert legacy_bad == (900.0, 500.0)
    assert invalid is None and invalid_info["click_out_of_bounds"]
    print("pipeline: OK")


if __name__ == "__main__":
    test_reward()
    test_adaptive_reward()
    test_normalized_bbox()
    test_vllm_engine_kwargs()
    test_dry_run_gate()
    test_pipeline()
    print("all GRPO tests passed")
