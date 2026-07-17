# RESULT_FORMAT.md — Windows → Mac 回传约定（一页纸）

## Zip 名

`C1_mirror_results.zip`

## 必选文件（每个 trial）

路径：`recordings/{id}/mirror/<任意子目录>/`

| 文件 | 用途 |
|------|------|
| `compare_metrics.csv` | 论文指标 |
| `compare_aligned.csv` | 对齐曲线/散点 |
| `video_va.csv` | 逐帧 VA |
| `summary.json` | valid_ratio 等 |

`{id}` = `001` … `020`（与录制一致）

## 建议额外

- `c1_summary.json`（全 trial 均值）
- `c1_all_metrics.csv`（拼表）
- `compare_curve.png`

## 不要改

- 原始 `video.mp4`、`source_va.csv`（可不放入 zip）

Mac 收到后解压到 `RAL/experiment/C1/result/`，跑 `run_c1_aggregate.py` + `plot_c1.py`。
