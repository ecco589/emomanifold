# EmotionManifold 路线图

## 里程碑

| 阶段 | 目标 | 状态 |
|------|------|------|
| M1 | 文字 → ERNIE 流式 → 逐 token VAD | ✓ |
| M2 | + Grimace 逐 token 表情 | ✓ |
| M3 | VA → FKCN → 29 DOF 实机表情 | ✓ |
| M4 | 逐 token 实机表情 | 待做 |
| M5 | 加入对话功能 | 待做 |

## M3（已完成）

1. VA→7 权重可视化（`M3test/FKCN_visualize/`）
2. 7 权重 → factor 融合 → 下发（`face/face_driver.py`）
3. 实机联调（`face/server.py` + `webui/`）

部署包：`face_m3.zip`，说明见 `face/README_DEPLOY.md`

## M4

ERNIE 流式生成，每个 token 的 VAD/FKCN 权重实时驱动实机。

## M5

加入对话功能。
