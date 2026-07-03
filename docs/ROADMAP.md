# EmotionManifold 路线图

## 里程碑

| 阶段 | 目标 | 状态 |
|------|------|------|
| M1 | 文字 → ERNIE 流式 → 逐 token VAD | ✓ |
| M2 | + Grimace 逐 token 表情 | ✓ |
| M3 | VA → FKCN → 29 DOF 实机表情 | ✓ |
| M4.0_RAL | Qwen 管线 + RAL 第一阶段实验（E1–E3） | ✓ |
| M4.1_RAL | 第二阶段实验（句级/token 投影）+ 撰稿 | 待做 |
| M4.2_RAL | 第三阶段（实机镜像测试）+ 定稿 | 待做 |
| M5 | 加入对话功能 | 待做 |

## M3（已完成，回溯见 tag `M3`）

1. 7 权重 → factor 融合 → 下发（`face/face_driver.py`）
2. 实机联调（`face/server.py` + `webui/`）

部署包：`face_m3.zip`，说明见 `face/README_DEPLOY.md`。  
需 M3 可视化工具时：`git checkout M3`（含 `M3test/FKCN_visualize/`）。

## M4.0_RAL（已完成，回溯见 tag `M4.0_RAL`）

**管线**：主模型切换为 **Qwen2.5-0.5B**；极词挖掘、AdaptiveScaler、流式投影对齐 NIPS B1；中文/英文 VAD 词库与 NRC 对齐。

**RAL 投稿准备**：`RAL/ieeeconf/` 模板 + `RAL/outline.md` 实验大纲。

**§IV.A 情感子空间可靠性（已完成）**

| 实验 | 内容 | 产出 |
|------|------|------|
| **E1** | 极词 N=10–80 sweep × Spearman（六模型 mean + min–max 带） | `RAL/experiment/E1/output/e1_spearman_pole_sweep.pdf` |
| **E2** | 跨模型 Spearman（NIPS Table 1 五模型 + Qwen 0.5B，pole=50） | `RAL/experiment/E2/output/e2_table.tex` |
| **E3** | EN / ZH 极词 × 轴正交 + 跨语情绪簇 vs 随机 3D 基线 | `RAL/experiment/E3/output/e3_table.tex` |

复现：`python RAL/experiment/E1/code/run_e1.py` → `export_pole_sweep_series.py` → `plot_e1.py`；E2/E3 见各目录 `README.md`。

## M4.1_RAL（下一步）

- §IV.B：EmoBank / GoEmotions / iSarcasmEval 句级与 token 级投影验证
- 正文 Methods + Experiments A 撰写

## M4.2_RAL

- §IV.C：实机镜像测试与人脸识别 round-trip
- 全文定稿投稿

## M5

加入对话功能。
