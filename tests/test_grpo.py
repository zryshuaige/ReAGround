# -*- coding: utf-8 -*-
"""CPU tests for region rewards and the two-call GRPO inference pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.build_grpo_dataset import normalized_bbox, region_target
from infer.pipeline import ground_grpo
from protocol import MAX_PIXELS
from train.grpo_reward import REWARD_FUNCS, parse_bbox
from train.grpo_reward import reward_metrics
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

    def fake_generate(model, processor, image, mode, instruction, max_new_tokens=64):
        calls.append((mode, image.size))
        if mode == "region":
            return "<bbox>[100,0,500,500]</bbox>", image.size
        return "(%d,%d)" % (image.width // 2, image.height // 2), image.size

    original = pipeline.generate
    pipeline.generate = fake_generate
    try:
        point, info = ground_grpo(None, None, Image.new("RGB", (1000, 500)), "open Python")
    finally:
        pipeline.generate = original
    assert point == (300.0, 125.0)
    assert calls == [("region", (1000, 500)), ("direct", (400, 250))]
    assert info["crop_box"] == (100, 0, 500, 250)
    print("pipeline: OK")


if __name__ == "__main__":
    test_reward()
    test_normalized_bbox()
    test_vllm_engine_kwargs()
    test_dry_run_gate()
    test_pipeline()
    print("all GRPO tests passed")
