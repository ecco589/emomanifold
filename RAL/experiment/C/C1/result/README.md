# C1 原始数据（result/）

论文主数据 = `multicond_noanchor_20260731`（20260731 无锚定四条件同场，80/80 含 Mirror）。

```
result/
├── multicond_noanchor_20260731/   四条件录像 + Mirror
│   ├── ours/ mishra_gpt/ xpress_gpt/ random/   各 001..020
│   └── _latency_labels/                        延迟标签侧记录
└── label_precompute/              Mishra/Xpress GPT 标签预跑（非 Mirror）
```

每条 trial 保留：`source_va.csv`、`session.json`、`mirror/*/video_va.csv`、`video.mp4`（gitignore）。

历史数据（20260717 旧场、20260730 锚定场、试录）已归档至
`RAL/experiment/_archive_obsolete_20260731/`。锚定场（β=0.8 句级原型锚定）
因执行端压扁句内效价动态而弃用，详见归档目录内的 summary。
