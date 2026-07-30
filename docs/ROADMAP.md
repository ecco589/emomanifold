# EmotionManifold 路线图

## 里程碑

| 阶段 | 目标 | 状态 |
|------|------|------|
| M1 | 文字 → ERNIE 流式 → 逐 token VAD | ✓ |
| M2 | + Grimace 逐 token 表情 | ✓ |
| M3 | VA → FKCN → 29 DOF 实机表情 | ✓ |
| M4.0_RAL | Qwen 管线 + RAL 子空间实验（A1–A3） | ✓ |
| M4.1_RAL | 情感细节追踪（B1）+ 撰稿 | 进行中 |
| M4.2_RAL | 实机 C1 Mirror + C2 人因 + 定稿 | 待做 |
| M5 | 加入对话功能 | 待做 |

## M3（已完成，回溯见 tag `M3`）

1. 7 权重 → factor 融合 → 下发（`face/face_driver.py`）
2. 实机联调（`face/server.py` + `webui/`）

部署包：`face_m3.zip`，说明见 `face/README_DEPLOY.md`。  
需 M3 可视化工具时：`git checkout M3`（含 `M3test/FKCN_visualize/`）。

## M4.0_RAL（已完成，回溯见 tag `M4.0_RAL`）

**管线**：主模型切换为 **Qwen2.5-0.5B**；极词挖掘、AdaptiveScaler、流式投影对齐 NIPS B1；中文/英文 VAD 词库与 NRC 对齐。

**RAL 投稿准备**：`RAL/ieeeconf/` 模板 + `RAL/outline.md` 实验大纲。

**§IV.A 子空间提取（已完成）** — 目录 `RAL/experiment/A/`

| 实验 | 内容 | 产出 |
|------|------|------|
| **A1** | 极词 N=10–80 sweep × Spearman（六模型 mean + min–max 带） | `RAL/experiment/A/A1/output/e1_spearman_pole_sweep.pdf` |
| **A2** | 跨模型 Spearman（NIPS Table 1 五模型 + Qwen 0.5B，pole=50） | `RAL/experiment/A/A2/output/e2_table.tex` |
| **A3** | EN / ZH 极词 × 轴正交 + 跨语情绪簇 vs 随机 3D 基线 | `RAL/experiment/A/A3/output/e3_table.tex` |

复现：`python RAL/experiment/A/A1/code/run_e1.py` → `export_pole_sweep_series.py` → `plot_e1.py`；A2/A3 见各目录 `README.md`。

## M4.1_RAL

- §IV.B 情感细节追踪：`RAL/experiment/B/B1`（语境化讽刺）
- 正文 Methods + Experiments 撰写

## M4.2_RAL

- §IV.C：`C/C1` Mirror Test + `C/C2` 人因（识别 + 2AFC）
- 全文定稿投稿

## M5

加入对话功能。
