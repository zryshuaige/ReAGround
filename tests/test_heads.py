# -*- coding: utf-8 -*-
"""Dense-head math tests; optional real-model forward/backward/save checks."""
import json
import os
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modeling_reaground import (ReAGroundHeadModel, decode_point, grid_targets,
                                head_loss)


def test_grid_targets_and_decode():
    for grid in [(1, 1), (3, 5), (8, 4)]:
        gh, gw = grid
        for bbox in ([0.0, 0.0, 0.001, 0.001],
                     [0.999, 0.999, 1.0, 1.0],
                     [0.101, 0.202, 0.109, 0.218]):
            region, index, offset = grid_targets(bbox, grid)
            assert region.shape == (gh * gw,)
            assert region.sum() >= 1
            assert region[index] == 1
            assert torch.all(offset.abs() <= 1)
            n = gh * gw
            scores = torch.full((n,), -100.0)
            scores[index] = 100.0
            raw_offsets = torch.zeros(n, 2)
            raw_offsets[index] = torch.atanh(offset.clamp(-0.999999, 0.999999))
            outputs = {"region_logits": [region], "point_scores": [scores],
                       "point_offsets": [raw_offsets], "grids": [grid]}
            point, info = decode_point(outputs, (1600, 900))
            expected = ((bbox[0] + bbox[2]) / 2 * 1600,
                        (bbox[1] + bbox[3]) / 2 * 900)
            assert abs(point[0] - expected[0]) < 0.01, (point, expected)
            assert abs(point[1] - expected[1]) < 0.01, (point, expected)
            assert info["point_cell"] == list(divmod(int(index), gw))
    print("grid targets and continuous decode: OK")


def test_loss_and_region_dependency():
    torch.manual_seed(0)
    region = torch.randn(6, requires_grad=True)
    point = torch.randn(6, requires_grad=True)
    offsets = torch.randn(6, 2, requires_grad=True)
    fused = point + region
    outputs = {"region_logits": [region], "point_scores": [fused],
               "point_offsets": [offsets], "grids": [(2, 3)]}
    loss, metrics = head_loss(outputs, torch.tensor([[0.2, 0.2, 0.4, 0.5]]))
    assert loss.isfinite() and all(value >= 0 for value in metrics.values())
    loss.backward()
    for tensor in (region, point, offsets):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0
    changed = point.detach() + 10 * region.detach()
    assert not torch.equal(fused.detach(), changed)
    print("joint loss, gradients and region dependency: OK")


def test_instruction_before_image():
    from protocol import build_messages
    content = build_messages("heads", "click Save")[0]["content"]
    assert [item["type"] for item in content] == ["text", "image"]
    assert "click Save" in content[0]["text"]
    print("instruction-first prompt: OK")


def test_real_model(model_path, image_path):
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor
    from protocol import build_messages, resized_dims
    from train.train_lora import LORA_TARGETS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = ReAGroundHeadModel.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((448, 448))
    resized = image.resize(resized_dims(image.width, image.height))
    prompt = processor.apply_chat_template(
        build_messages("heads", "click the target"), tokenize=False,
        add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[resized], return_tensors="pt").to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs)
    grid = outputs["grids"][0]
    t, h, w = (int(v) for v in inputs["image_grid_thw"][0])
    merge = model.config.vision_config.spatial_merge_size
    assert t == 1 and grid == (h // merge, w // merge)
    assert outputs["region_logits"][0].shape == (grid[0] * grid[1],)
    assert outputs["point_offsets"][0].shape == (grid[0] * grid[1], 2)
    point, _ = decode_point(outputs, image.size)
    assert 0 <= point[0] <= image.width and 0 <= point[1] <= image.height
    from infer.pipeline import ground_heads
    pipeline_point, pipeline_info = ground_heads(
        model, processor, image, "click the target")
    assert pipeline_point == point and pipeline_info["grid"] == list(grid)
    print("real single forward and ground_heads pipeline: OK",
          {"grid": grid, "point": point})

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=LORA_TARGETS,
                      modules_to_save=["region_head", "point_head"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    outputs = model(**inputs)
    gt = torch.tensor([[0.2, 0.2, 0.4, 0.4]], device=device)
    loss, _ = head_loss(outputs, gt)
    loss.backward()
    for name in ("region_head", "point_head"):
        params = [p for n, p in model.named_parameters()
                  if name in n and p.requires_grad]
        assert params and all(p.grad is not None and p.grad.abs().sum() > 0
                              for p in params), name
    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    assert lora_params and any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in lora_params)
    print("real backward through both heads and LoRA: OK", float(loss))

    model.eval()
    with torch.inference_mode():
        before = model(**inputs)
    with tempfile.TemporaryDirectory() as output_dir:
        model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "head_config.json"), "w") as f:
            json.dump({"region_layer": model.base_model.model.region_layer}, f)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        from protocol import load_head_model
        restored, _ = load_head_model(model_path, output_dir)
        with torch.inference_mode():
            after = restored(**inputs)
    for key in ("region_logits", "point_scores", "point_offsets"):
        torch.testing.assert_close(before[key][0].cpu(), after[key][0].cpu(),
                                   rtol=0, atol=0)
    print("adapter plus both heads save/load round-trip: OK")


if __name__ == "__main__":
    test_grid_targets_and_decode()
    test_loss_and_region_dependency()
    test_instruction_before_image()
    model = os.environ.get("REAGROUND_MODEL")
    image = os.environ.get("REAGROUND_TEST_IMAGE")
    if model and image:
        test_real_model(model, image)
    else:
        print("skip real model test (set REAGROUND_MODEL and REAGROUND_TEST_IMAGE)")
    print("all head tests passed")
