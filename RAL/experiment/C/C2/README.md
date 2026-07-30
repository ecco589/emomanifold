# C2 人类问卷

## 目录

```
C2/
├── data/                    刺激模板（非录像）
│   ├── t1_prompts.csv       Task1 认表情（10 条）
│   └── t2_prompts.csv       Task2 转折句（10 条）
├── result/
│   └── session_noanchor_20260731/    实机同场录像（无锚定，与 C1 主数据同场）
└── output/                  （待做）裁切后的问卷材料包
```

## Task 说明

- **T1**：七选一认表情 → `session_.../t1/ours/T1-xx/video.mp4`
- **T2**：token vs turn vs random 二选一 → `t2/ours`（token）、`t2/mishra_gpt`、`t2/xpress_gpt`（turn 静脸）、`t2/random`

**待做：** 从 `result/session_noanchor_20260731/` 裁切 900×1080 无声视频 → `output/survey/`。

已归档：20260730 锚定场（`RAL/experiment/_archive_obsolete_20260731/C2/`）；更早的 7/23 旧 survey。
