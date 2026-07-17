# C1 Windows Mirror 分析（交接给 Windows Cursor）

## 背景

Mac 上跑 Py-Feat 不顺（`torchcodec` 导入挂起、HuggingFace 直连超时）。  
**Windows 上该套 `video_va_system` 已验证可顺畅运行**，因此分析改在 Windows 完成。

工控机录制已完成：`result/recordings/{001..020}/` 各含 `video.mp4` + `source_va.csv`。

---

## 你的任务（Windows Cursor）

在本目录实现/跑通批处理，对 **20 条** trial 做 Mirror 分析并打包回传。

### 输入（已有，从 Mac 拷过来）

```
recordings/
├── 001/
│   ├── video.mp4
│   ├── source_va.csv
│   └── session.json          # 可选
├── 002/
│   └── ...
└── 020/
```

`source_va.csv` 列（理想 VA，token 级）：

```csv
timestamp_s,valence,arousal,token
6.744,0.9009,0.3479,Win
...
```

（`token` 列可忽略；对齐只用 `timestamp_s,valence,arousal`）

### 工具

复用仓库内：

- `../code/mirror/`（即原 `video_va_system`：`video_va_pipeline.py`、`setup_env.bat`、`run.bat`）
- 或本机已装好的同一套环境

首次：

```bat
cd ..\code\mirror
setup_env.bat
diagnose.bat
```

### 建议实现

新建本目录脚本，例如 `run_c1_windows_batch.py` / `.bat`：

1. 遍历 `../result/recordings/*/`
2. 对每个 trial 调用：

```bat
python video_va_pipeline.py ^
  --video ..\result\recordings\001\video.mp4 ^
  --source-va ..\result\recordings\001\source_va.csv ^
  --out-dir ..\result\recordings\001\mirror ^
  --time-offset-ms 150 ^
  --device auto
```

3. 终端打印进度：`[C1] 3/20 analyzing 003 ...`
4. 支持 `--skip-existing`（已有 `compare_metrics.csv` 则跳过）
5. 全部完成后运行汇总（可调用 `../code/run_c1_aggregate.py`），或自写同等汇总

`--time-offset-ms`：默认 **150**；若曲线整体错位，可试 100–300。

---

## 必须回传的结果格式（约定）

把下面结构打成 zip（例如 `C1_mirror_results.zip`）交给 Mac 侧即可。

### 目录结构（必须）

```
C1_mirror_results/
├── recordings/
│   ├── 001/
│   │   └── mirror/
│   │       └── <任意时间戳子目录>/     # video_va_pipeline 默认行为
│   │           ├── video_va.csv              # 必选：逐帧 VA
│   │           ├── compare_aligned.csv       # 必选：与 source 对齐
│   │           ├── compare_metrics.csv       # 必选：指标
│   │           ├── compare_curve.png         # 建议
│   │           ├── video_va_curve.png        # 建议
│   │           └── summary.json              # 必选：含 valid_ratio
│   ├── 002/
│   │   └── mirror/...
│   └── ...
├── c1_all_metrics.csv      # 建议：所有 trial 的 metrics 拼表
└── c1_summary.json         # 建议：均值 Pearson/MAE/RMSE/CCC
```

### `compare_metrics.csv` 列（pipeline 已生成，勿改名）

```csv
metric,n,mae,rmse,mean_error,pearson_r,ccc
valence,...
arousal,...
va_distance,...
```

### `compare_aligned.csv` 关键列

至少含：`aligned_timestamp_s`（或 source 时间列）、`source_valence`、`source_arousal`、`detected_valence`、`detected_arousal`

### `c1_summary.json` 示例

```json
{
  "n_trials": 20,
  "time_offset_ms": 150,
  "metrics": {
    "valence":  {"pearson_r_mean": 0.0, "mae_mean": 0.0, "rmse_mean": 0.0, "ccc_mean": 0.0},
    "arousal":  {"pearson_r_mean": 0.0, "mae_mean": 0.0, "rmse_mean": 0.0, "ccc_mean": 0.0},
    "va_distance": {"mae_mean": 0.0, "rmse_mean": 0.0}
  }
}
```

---

## 验收清单

- [ ] 20/20 trial 都有 `compare_metrics.csv`
- [ ] 各 trial `summary.json` 里 `valid_ratio` 合理（机器人脸建议尽量 >0.5；过低在报告里注明）
- [ ] zip 解压后路径符合上文结构
- [ ] Mac 侧可直接跑：`python ../code/run_c1_aggregate.py --recordings ../result/recordings`（若你把 mirror 结果写回同结构）

---

## 注意

- **不要**改 `source_va.csv` / `video.mp4` 内容
- 首次跑 Py-Feat 会下模型权重，需能访问 HuggingFace（或已缓存）
- 机器人脸可能 valid_ratio 偏低，属预期风险，照常出数即可
- 超过 30 秒无终端输出视为卡住，应打印心跳/进度

---

## 与 Mac 的分工

| 侧 | 职责 |
|----|------|
| 工控机 | 已完成录制 |
| **Windows（你）** | Mirror 批分析 + 按本约定打包 |
| Mac | 收 zip → 汇总图/写论文 §IV-C |
