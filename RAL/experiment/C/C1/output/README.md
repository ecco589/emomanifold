# C1 分析输出（仅图表与论文材料）

`output/` **只放给人看的分析结果**（图、表、caption）。原始 JSON/CSV 在 `../result/`。

```
output/
└── multicond_noanchor_20260731/   ← 论文主结果（20260731 无锚定四条件同场）
    ├── paper/                     主图 PDF/PNG + caption（fig:c1_main）
    │   ├── c1_main_figure.pdf
    │   ├── c1_main_caption.md / .tex
    ├── c1_multicond_summary.md    四条件对比表（ours V 87.5% / A 69.6%）
    ├── c1_multicond_summary.csv
    └── c1_multicond_full.json
```

SOTA 标签预跑（非 Mirror 分析）在 `../result/label_precompute/`。
历史输出（旧场 77.8% 主数字、锚定场对照表）在 `RAL/experiment/_archive_obsolete_20260731/`。

复现：

```bash
python RAL/experiment/C/C1/code/run_c1_multicond_compare.py    # → 对照表
python RAL/experiment/C/C1/code/plot_c1_multicond_paper.py     # → paper/ 主图
```
