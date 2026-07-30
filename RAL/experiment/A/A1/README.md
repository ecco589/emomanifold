# A1：Qwen 梯度 Spearman（极词 N=10–80，中/英）

## 流程（对齐 NIPS B1）

1. **Hold-out（Spearman）**
   - **英文**：NIPS 五模型 tokenizer 交集 300 词（`data/nips_holdout_intersection_en.json`），与 fig2 阴影带同一词表
   - **中文**：Qwen 单 token hold-out（`build_holdout.py` → `result/holdout_zh.json`）
2. **极词 sweep**：N = 10, 20, …, 80（每极 N 个，六极）；**不排除** hold-out（与 NIPS 一致）
3. **轴**：embedding 簇差分 → V/A/D
4. **Scaler**：极词语料训练 + `enforce_nonnegative_scales`
5. **Spearman**：hold-out 词 embedding 投影 vs NRC（bipolar [-1,1]）

## 运行

```bash
# 1. 中文 hold-out（英文用 data/nips_holdout_intersection_en.json，无需 build）
python RAL/experiment/A/A1/code/build_holdout.py

# 2. 全 sweep（约 30–60 分钟，CPU）
python RAL/experiment/A/A1/code/run_e1.py

# 3. 出图（NIPS fig2 版式：六模型 mean 实线 + min–max 带）
python RAL/experiment/A/A1/code/export_pole_sweep_series.py
python RAL/experiment/A/A1/code/plot_e1.py
```

## 目录

| 路径 | 内容 |
|------|------|
| `data/nips_holdout_intersection_en.json` | NIPS 五模型交集 Spearman 词表（英文） |
| `result/holdout_zh.json` | 中文 Spearman 词表 |
| `result/{en,zh}/pole_{N}/` | 各档 report.json |
| `data/nips_b1_per_model_spearman.json` | NIPS 五模型逐档 Spearman |
| `result/pole_sweep_mean_rho.json` | 六模型 mean / min / max 汇总 |
| `output/e1_spearman_pole_sweep.png` | 六模型 mean 线 + min–max 带（V/A/D） |

## 说明

- Qwen2.5-0.5B 对 NIPS 交集 300 词均为单 token；非单 token 词在 Spearman 中自动跳过。
- 英文 N=80 时若某极 Qwen 单 token 不足 80，summary 会记录 error。
