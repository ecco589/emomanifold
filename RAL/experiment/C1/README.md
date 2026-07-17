# C1 实机 Mirror Test

## 分工（当前）

| 位置 | 做什么 |
|------|--------|
| **工控机** | 录制：`result/recordings/{id}/video.mp4` + `source_va.csv`（已完成） |
| **Windows** | Mirror 分析（Py-Feat）→ 见 [`windows/README.md`](windows/README.md) |
| **Mac** | 收 Windows 回传 zip → 汇总出图 / 写论文 |

> Mac 上 Py-Feat/`torchcodec`/HF 下载不稳；Windows 上原 `video_va_system` 已跑通，分析改在 Windows。

---

## Windows 侧（你要做的）

1. 把 `C1/`（至少含 `result/recordings/` + `code/mirror/` + `windows/`）拷到 Windows
2. 打开 Cursor，按 **`windows/README.md`** 实现批处理并跑完 20 条
3. 按该文档约定的目录结构打成 `C1_mirror_results.zip` 传回 Mac

回传约定摘要：

```
C1_mirror_results/recordings/{id}/mirror/<run>/
  video_va.csv
  compare_aligned.csv
  compare_metrics.csv
  summary.json
```

---

## Mac 侧（收到 zip 后）

```bash
# 1) 把 zip 里 recordings/*/mirror/ 同步到 result/recordings/
# 2) 再分析（滞后 + 平滑 + 去偏）——论文主结果
python RAL/experiment/C1/code/run_c1_reanalyze.py

# 可选：固定滞后敏感性
python RAL/experiment/C1/code/run_c1_reanalyze.py --fixed-lag 0.8 \
  --out-dir RAL/experiment/C1/output/reanalysis_fixed800 \
  --result-dir RAL/experiment/C1/result/reanalysis_fixed800
```

---

## 目录

```
C1/
├── README.md
├── ROBOT_REQUIREMENTS.md
├── windows/README.md
├── data/stimuli*.csv
├── code/
│   ├── mirror/                 # Py-Feat 管线
│   └── run_c1_reanalyze.py     # 校正后再分析（主）
├── output/
│   ├── reanalysis_best_lag/    # 论文主图/主表
│   └── reanalysis_fixed800/    # 固定 800 ms 敏感性
└── result/
    ├── recordings/{id}/        # 录制 + mirror/
    ├── reanalysis_best_lag/    # JSON/CSV 主结果
    ├── reanalysis_fixed800/
    └── windows_raw/            # Windows 原始汇总（未校正）
```

## 指标

- Windows 原始：`compare_metrics.csv`（固定 +150 ms）
- 论文主报：`output/reanalysis_best_lag/` 中 L2/L3 Pearson（滞后+平滑+去偏）
