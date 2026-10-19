# ReAGround 首轮受控实验 — 简报（2026-10-18，更新版）

## 总结论

闸门 1–7 已全部走完（7a ZonUI 按用户指示跳过；7b 基座实测偏离预期，已完整记录）。
训练管线代码缺陷经用户授权修复后，闸门 5 复测通过。因 A10 实测训练速度仅为方案
预估的 1/3.7（180s/步 vs 48s/步），两组全跑将严重超过 24 GPU·h 预算，
**用户决策：两组训练均暂不执行**（group A 启动后在第 4/332 步终止，GPU 已释放）。

## 代码改动记录（用户授权修 bug + 两项加速，未偏离实验方案其余部分）

**改动 1（bug 修复）**：`train/dataset.py` 第 101 行
```diff
-            "pixel_values": enc["pixel_values"][0],
+            "pixel_values": enc["pixel_values"],  # already flat [num_patches, dim]; no batch dim
```
原因：Qwen2.5-VL processor 返回的 `pixel_values` 无 batch 维（按 patch 拉平的
`[num_patches, 1176]`），`[0]` 只保留第 0 个 patch，与 `image_grid_thw` 声明的
网格不符，导致视觉编码器 reshape 崩溃。`image_grid_thw` 的 `[0]` 保留不动
（它有 batch 维 `[1,3]`，取第 0 张图正确）。

**改动 2（加速：FA2，用户指定方案 1）**：`train/train_lora.py` 第 59 行
`attn_implementation="sdpa"` → `"flash_attention_2"`。
环境新增：flash-attn 2.5.6 预编译 wheel（cu122torch2.1cxx11abiFALSE-cp310，
经 gh-proxy 下载），已在 A10（sm_86）上验证 FA2 kernel 真实可用。
注：评测侧 `eval/eval_screenspot.py --attn` 保持默认 sdpa，未动。

**改动 3（加速：epochs 减半，用户指定方案 2）**：
`configs/train_lora.yaml` 与 `configs/train_group_a.yaml` 的 `epochs: 2` → `epochs: 1`。

**改动 4（中途存档，用户指定）**：`train/train_lora.py` TrainingArguments
`save_strategy="epoch"` → `save_strategy="steps", save_steps=200`。
group D 训练将在第 200 步保存 `outputs/reaground_lora/checkpoint-200/`
（含 optimizer 与 trainer_state/逐步 loss 历史）后继续至 333 步；
最终 adapter 仍由脚本末尾 `save_pretrained` 落盘。该改动在 group A 训练
启动后落盘，不影响已加载旧代码运行中的 A。

修复后验证（FA2 + epochs 1，两配置各 dry-run 2 步，退出码均 0）：
- twostage：loss 0.652，2 步 26.1s（原 sdpa 41.9s → **1.61×**）
- direct：loss 0.333，2 步 31.3s（→ **1.34×**）
- 张量形状核验：pixel_values (4888, 1176) 与 grid [1,52,94] 一致

其余环境修补（非代码）：torchvision 0.21.0+cu124 → 0.16.2（匹配 torch 2.1.2）；
语料下载改走 ModelScope（hf-mirror xet 桥 403）；showui_web 解压产物上移一层
以匹配代码期望的路径布局（纯数据布局调整）。

**加速后训练耗时估计**：按 dry-run 实测加速比推算，epochs=1 时
group A ≈ 166 步 × 110–135s ≈ 5–6h；group D ≈ 333 步 ≈ 10–12.5h
（正式跑长序列的每步时间以实际日志为准）。

## 各闸门结果

| 闸门 | 结果 | 说明 |
|---|---|---|
| 1 环境+权重+benchmark | ✅ 通过 | 中途 HF 断线重试 1 次；模型 22G + benchmark 3.4G |
| 2 CPU 单测 | ✅ 通过 | torchvision 降级后 `all tests passed`（含扩展测试） |
| 3 训练语料 | ✅ 通过（有修补） | web 经 ModelScope 续传解压；desktop 150 文件 0.52G |
| 4 语料 schema | ✅ 通过（有修补） | web 599,186 条 + desktop 7,496 条，样例正常 |
| 5 训练冒烟 | ✅ 通过（修 bug 后） | 初测因 dataset.py 缺陷失败；修复后 2 步 loss 0.654 |
| 6 评测数据 | ✅ 通过 | pro 1581 条 / v2 1272 条，missing images = 0 |
| 7a ZonUI ≈29% | ⏭ 跳过 | 用户指示"先别测 ZonUI-3B" |
| 7b 基座 ≈16% | ❌ 明显偏离 | 实测 **1.5%**（23/1581），见下文"已记录的偏差" |
| 8 训练组 A/D | ⏸ 暂不执行 | 用户决策（预算原因），A 组在第 4/332 步被终止 |

## 已记录的偏差：基座金标准（闸门 7b）

`outputs/base_pro.jsonl`（全量 1581 条已保存）：

```
overall: 23/1581 (1.5%)        ← 预期 ≈16%
  text 22/977 (2.3%)   icon 1/604 (0.2%)
  macos 11/604 (1.8%)  windows 12/927 (1.3%)  linux 0/50 (0.0%)
parse failures (counted as miss): 1000/1581 (63%)
```

偏离主因：direct 提示词（protocol.py，逐字复制 ZonUI-3B 模型卡）要求直接输出坐标，
未做接地训练的基座 63% 样本输出自然语言散文。抽查原始输出确认权重与推理正常。
终端输出：`logs/07_base_pro.log`。

## 训练耗时实测（供修订方案参考）

- A10（150W 功耗墙，SM 1365/1695 MHz，util 91–100%，纯算力瓶颈）：
  group A 实测 180s/步 × 332 步 ≈ 16.6h（方案估 4.5h）；
  group D 16000 序列，预计 ≥ 19h。
- 评测速度参考：Pro 1581 条约 1.1h（~24 条/min）。

## 交付物清单

- 日志：`logs/` 下全部（setup、单测、语料下载、schema、dry-run 失败/修复后、
  评测 dry-run、基座评测、group A 启动日志）
- 结果文件：`outputs/base_pro.jsonl`（基座 Pro 全量 1581 条）
- 数据产物：`data/outputs/train_twostage.jsonl`（16000 序列）+
  `train_direct.jsonl` + `val.jsonl`（250）
- 失败分析：`logs/05_GATE5_FAILURE.md`
- 未产生：`zonui_pro.jsonl`（跳过）、`group_a/d_*`（未训练）、
  `outputs/group_a/`、`outputs/reaground_lora/`

## 实际耗时

墙钟约 6.5 h；GPU 消耗约 1.6 h（基座评测 1.1h + 冒烟/试跑 0.5h），预算内。

## 待办（等用户指示）

1. 训练组 A/D 是否启动：FA2 + epochs=1 后 A ≈ 5–6h、D ≈ 10–12.5h（均含评测另加 ~2h/组），
   两组合计约 19–23h，基本落入 24 GPU·h 预算；两条数据通路 dry-run 均已验证通过，
   启动命令 `bash scripts/train_group_a.sh` / `bash scripts/train_group_d.sh`；
2. 是否补跑 ZonUI 金标准（约 1h）以交叉验证评测管线，进而定性基座 1.5% 的偏差；
3. 全部改动见上文"代码改动记录"（3 处代码/配置 + 环境修补），数据与随机种子未动。
