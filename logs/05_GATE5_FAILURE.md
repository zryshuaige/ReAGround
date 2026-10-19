# 闸门 5 失败记录：训练管线冒烟（train_lora.py --dry-run）

时间：2026-10-18 ~08:55 (UTC+8 服务器时间)
状态：**未通过**。按红线（不改代码）停止训练侧流程，完整记录如下。

## 现象

`python train/train_lora.py --config configs/train_lora.yaml --dry-run` 在第 1 个训练步崩溃：

```
RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280
  File "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 529, in forward
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, ...)
```

完整 traceback 见 `logs/05b_train_dryrun.log`。模型加载、LoRA 注入（18,576,384 trainable params, 0.49%）、数据加载均正常，失败发生在视觉编码器前向。

## 根因（已通过最小复现确认，未修改任何代码）

`train/dataset.py` 中 `GroundingSFT.__getitem__`：

```python
"pixel_values": enc["pixel_values"][0],
"image_grid_thw": enc["image_grid_thw"][0],
```

Qwen2.5-VL 的 processor（transformers 4.49.0，slow image processor）返回的
`pixel_values` 是**按 patch 拉平**的二维张量 `[num_patches, 1176]`（batch 维已被合并），
而 `image_grid_thw` 是 `[num_images, 3]`。

- `enc["image_grid_thw"][0]` → `[t,h,w]` 取第 0 张图的网格：正确。
- `enc["pixel_values"][0]` → 只取到**第 0 个 patch**（形状 `(1176,)`）：错误。

实测证据（未改代码，直接调用数据集类）：

```
{'pixel_values': (1176,), 'image_grid_thw': (3,)}   # grid = [1, 52, 94] = 4888 patches
```

即 grid 声明 4888 个 patch，实际只传入 1 个。经 patch_embed 后 hidden_states 为
`[1, 1280]`，`reshape(1 // 4, 4, -1)` = `reshape(0, 4, -1)` → 报上述错误。

- 与 batch 内容无关：前 8 条样本图片尺寸逐一检查均正常（最小 476×532）。
- 与 peft/transformers 兼容性无关：单测 `test_core.py` 全过（含 REAGROUND_MODEL
  扩展测试：`tokenize_target: OK`），但单测未覆盖 `__getitem__` 的像素路径。
- 评测路径（`eval/eval_screenspot.py` → `protocol.generate`）不使用该索引写法，
  不受影响。

## 影响

- 训练入口（group A / group D）在修复前无法运行——任何样本都会在第 1 步崩溃。
- 修复方向（供作者参考，**未执行**）：`pixel_values` 不应取 `[0]`，应保留完整
  `[num_patches, 1176]`；`collate` 中的 `torch.cat` 已是按拉平格式设计的。

## 环境（供复现）

torch 2.1.2+cu121, transformers 4.49.0, peft 0.17.1, accelerate 1.10.1,
torchvision 0.16.2（由 0.21.0+cu124 降级以匹配 torch 2.1.2）, Python 3.10.11, A10 24GB。
