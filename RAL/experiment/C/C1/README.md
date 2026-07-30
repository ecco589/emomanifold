# C1 实机 Mirror Test

## 两场数据

| 场次 | 原始数据 `result/` | 分析输出 `output/` |
|------|-------------------|---------------------|
| **legacy**（论文主数字） | `recordings/` | `legacy/`（paper + event + reanalysis） |
| **multicond**（四条件对照） | `multicond_20260730/` | `multicond_20260730/` |

SOTA 标签预跑（开发机中间数据）→ `result/label_precompute/`。

## 复现

```bash
python RAL/experiment/C/C1/code/run_c1_event_analysis.py
python RAL/experiment/C/C1/code/run_c1_reanalyze.py
python RAL/experiment/C/C1/code/run_c1_multicond_compare.py
```

## 目录

```
C1/
├── data/stimuli.csv
├── code/                     # 分析 + baselines/（Mishra/Xpress）
├── handoff/                  # 实机交接
├── result/                   # 原始录像、Mirror CSV、label 预跑
└── output/                   # 仅图表与论文材料（见 output/README.md）
```
