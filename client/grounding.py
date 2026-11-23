import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

MODULE_DIR = Path(__file__).resolve().parent
FROZEN = getattr(sys, "frozen", False)
WORK_DIR = Path(sys.executable).resolve().parent if FROZEN else MODULE_DIR
if not FROZEN:
    REAGROUND = MODULE_DIR.parent
    if not (REAGROUND / "protocol.py").is_file():
        REAGROUND = REAGROUND / "ReAGround"
    if str(REAGROUND) not in sys.path:
        sys.path.insert(0, str(REAGROUND))

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from infer.parse import parse_point
from protocol import generate


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    workspace = os.environ.get("REAGROUND_WORKSPACE")
    directories = ((Path(workspace),) if workspace else ()) + (WORK_DIR,) + tuple(WORK_DIR.parents)
    for directory in directories:
        candidate = (directory / path).resolve()
        if candidate.exists():
            return candidate
    return (WORK_DIR / path).resolve()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    base: Path
    adapter: Optional[Path]
    results: Optional[Path]
    visual_tokens: Optional[int]

    @property
    def available(self):
        return self.adapter is None or (self.adapter / "adapter_model.safetensors").is_file()


@dataclass(frozen=True)
class Prediction:
    raw: str
    x: float
    y: float
    resized_wh: Tuple[int, int]


def load_specs():
    external = WORK_DIR / "models.json"
    config_path = external if FROZEN and external.is_file() else MODULE_DIR / "models.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = resolve_path(config["base"])
    specs = []
    for item in config["models"]:
        adapter = item.get("adapter")
        results = item.get("results")
        specs.append(ModelSpec(
            item["id"], item["name"], base,
            resolve_path(adapter) if adapter else None,
            resolve_path(results) if results else None,
            item.get("visualTokens"),
        ))
    return specs


class Grounder:
    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        if not self.spec.base.is_dir():
            raise FileNotFoundError(f"找不到基座模型：{self.spec.base}")
        if not self.spec.available:
            raise FileNotFoundError(
                f"找不到 {self.spec.name} 的 adapter_model.safetensors：{self.spec.adapter}\n"
                "请放入权重，或修改 client/models.json 中的路径。"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.spec.base), torch_dtype=dtype, device_map=device,
            attn_implementation="sdpa"
        )
        if self.spec.adapter:
            self.model = PeftModel.from_pretrained(
                self.model, str(self.spec.adapter)
            )
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.processor = AutoProcessor.from_pretrained(str(self.spec.base))

    def predict(self, image, instruction, visual_tokens=1280):
        self.load()
        raw, resized_wh = generate(
            self.model, self.processor, image, "direct", instruction,
            visual_tokens=visual_tokens)
        point = parse_point(raw)
        if point is None:
            raise ValueError(f"模型没有输出可解析的坐标：{raw}")
        rw, rh = resized_wh
        x, y = point[0] / rw * image.width, point[1] / rh * image.height
        if not (0 <= x < image.width and 0 <= y < image.height):
            raise ValueError(f"模型坐标超出截图：{raw}")
        return Prediction(raw, x, y, resized_wh)

    def close(self):
        self.model = None
        self.processor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
