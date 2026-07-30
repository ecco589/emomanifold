# C 系列实验总览（更新于 2026-07-30）

```
experiment/C/
├── README.md
├── C1/     Mirror + 四条件对照
├── C2/     人类问卷
└── C3/     Grimace 示意
```

## C1

| | legacy (20260717) | multicond (20260730) |
|--|--|--|
| **result/** | `legacy/recordings/` | `multicond_20260730/` |
| **output/** | `legacy/`（paper+event+reanalysis） | `multicond_20260730/`（对照表） |
| 中间数据 | — | `label_precompute/`（GPT 标签） |

## C2

| | 内容 |
|--|------|
| **data/** | T1/T2 刺激模板 CSV |
| **result/** | `session_20260730/` 实机 50 段录像 |
| **output/** | 待生成问卷裁切包 |

## 已清理（全 C 系列）

- C1：Mirror 中间文件 ~627MB；`_latency_labels`；Qwen 标签；result 收拢为 `legacy/`
- C2：旧 survey 归档 103MB；`t2/runs` 44MB；selection/zip
- 顶层 `C/baselines/`、`robot_handoff/`（前序整理）

论文引用速查见各子目录 README。
