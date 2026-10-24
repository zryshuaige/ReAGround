# -*- coding: utf-8 -*-
"""Shared protocol for ReAGround: prompts, output tags, resize budget, model I/O.

Everything that must stay identical between training and inference lives here.
The direct-mode prompt and the resize budget replicate the ZonUI-3B evaluation
protocol (https://huggingface.co/zonghanHZH/ZonUI-3B) exactly, so that our
evaluations of the base model, ZonUI-3B, and ReAGround share one protocol.
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# Resize budget (ZonUI-3B eval protocol). Qwen2.5-VL resizes every input image
# to a multiple of PATCH_FACTOR with total pixels inside [MIN_PIXELS, MAX_PIXELS];
# the model then speaks coordinates in that *resized* frame.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PATCH_FACTOR = 28  # patch_size 14 * merge_size 2

# Direct grounding prompt: verbatim from the ZonUI-3B model card.
PROMPT_DIRECT = ("Based on the screenshot of the page, I give a text description "
                 "and you give its corresponding location. The coordinate "
                 "represents a clickable location [x, y] for an element.")

# ReAGround two-round prompts. {instruction} is filled per sample.
PROMPT_ROUND1 = ("In this screenshot, locate the UI element for: '{instruction}'. "
                 "First, reason about which functional area of the interface it "
                 "belongs to based on the overall layout. Then output the area.")
PROMPT_ROUND2 = ("Within this cropped region, give the exact click point "
                 "for: '{instruction}'.")
PROMPT_REGION = ("To {instruction}, which screen region should be used? "
                 "Output only <bbox>[x1,y1,x2,y2]</bbox> with coordinates "
                 "normalized to [0,1000].")

PROMPTS = {"direct": PROMPT_DIRECT, "round1": PROMPT_ROUND1,
           "round2": PROMPT_ROUND2, "region": PROMPT_REGION}

# Output tags of the two-stage protocol.
TAGS = {"region": ("<region>", "</region>"),
        "area": ("<area>", "</area>"), "point": ("<point>", "</point>")}


def load_model(model_path, adapter_path=None, attn_implementation="sdpa"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_implementation)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.generation_config.do_sample = False
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def resized_dims(width, height):
    """smart_resize output as (resized_width, resized_height)."""
    rh, rw = smart_resize(height, width, factor=PATCH_FACTOR,
                          min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return rw, rh


def build_messages(mode, instruction):
    """One message constructor for both training and inference (never drift apart).

    Direct mode replicates the ZonUI-3B model card layout exactly: system text,
    image, then the query as a separate trailing text block.
    """
    if mode == "direct":
        return [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_DIRECT},
            {"type": "image", "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": instruction},
        ]}]
    return [{"role": "user", "content": [
        {"type": "text", "text": PROMPTS[mode].format(instruction=instruction)},
        {"type": "image", "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    ]}]


def generate(model, processor, image, mode, instruction, max_new_tokens=64):
    """Run one greedy generation on a PIL image. Returns (text, (resized_w, resized_h))."""
    rw, rh = resized_dims(image.width, image.height)
    resized = image.resize((rw, rh))
    text = processor.apply_chat_template(
        build_messages(mode, instruction), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[resized], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip(), (rw, rh)
