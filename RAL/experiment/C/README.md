# C 实机与人因实验

| 目录 | 内容 | 论文角色 |
|------|------|----------|
| **C1** | Mirror 闭环 + 四条件 SOTA 对照 | 主实验：句内转折执行保真度 |
| **C2** | 人类问卷（认表情 + 2AFC） | 补充：主观识别 |
| **C3** | Grimace 迁移示意 | 叙述：可部署性 |

## 统一约定

每个 C* 实验尽量遵循：

```
C*/
├── data/          刺激句 / 模板（小文件，入库）
├── code/          脚本（C1 含 baselines/）
├── handoff/       仅 C1：实机交接文档
├── result/        原始录像 + Mirror CSV + 中间 JSON
└── output/        图表、论文材料、分析表（给人看）
```

## 体积与清理原则（2026-07-30）

- **保留**：分析必需的 `video_va.csv`、`source_va.csv`、`session.json`
- **删除**：Mirror 管线中间产物（`video_va_full`、`compare_*`、曲线 PNG）、作废场次、Qwen 塌缩标签、重复归档

总览见 [`../STRUCTURE.md`](../STRUCTURE.md)。
