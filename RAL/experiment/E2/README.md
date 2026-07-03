# E2：跨模型 Spearman（NIPS Table 1 + Qwen2.5-0.5B）

固定 **pole_n=50**，复现 NIPS B1 五模型 Spearman，并加入本工作 **Qwen2.5-0.5B**（E1 `en/pole_50`）。

## 运行

```bash
python RAL/experiment/E2/code/run_e2.py
```

## 输出

| 路径 | 内容 |
|------|------|
| `data/table1_pole50.json` | 六模型 ρ_V/ρ_A/ρ_D |
| `output/e2_table.tex` | RAL/IEEE 表格 |
| `output/e2_spearman_table.md` | Markdown 表 |
| `output/e2_description.txt` | 实验说明 |

## 数据来源

- NIPS 五模型：`../E1/data/nips_b1_per_model_spearman.json`（pole 50）
- Qwen2.5-0.5B：`../E1/result/en/pole_50/report.json`

与 **E1**（极词 sweep）同为 Spearman 系列实验，见 `RAL/outline.md` §IV.A。
