# C2 人类问卷

人因在**独立发放的问卷**上完成（非本机录像包）。本目录只保留刺激模板、打包脚本与打分结果。

```
C2/
├── data/                         刺激模板（t1/t2 prompts）
├── code/build_survey_pack.py     历史裁切脚本（本机 session 已删，一般不用）
└── output/human_study_20260731/  ← 论文用结果
    ├── c2_scores_n20.xlsx
    ├── c2_summary.md
    └── c2_summary.json
```

## 协议（终稿）

- **T1**：模型按目标情绪回复 → 七选一认表情；另有预设标准表情上限对照。
- **T2**：**单选** token / turn / random（哪张脸更贴句内情感变化）。

## 主数字（n=20）

| | Ours | 对照 |
|--|--|--|
| T1 正确率 | 85.5% | 预设 90.5% |
| T2 选中率 | 83.5% | random 11.5% / turn 5.0% |

本机曾有的 `result/session_noanchor_20260731` 与 `output/survey`（未用于最终人因）已删除。
