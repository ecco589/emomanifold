# EmotionManifold 路线图

## 里程碑

| 阶段 | 目标 | 状态 |
|------|------|------|
| M1 | 文字 → ERNIE 流式 → 逐 token VAD | ✓ |
| M2 | + Grimace 逐 token 表情 | ✓ |
| M3 | VA → FKCN → 29 DOF 实机表情 | ✓ |
| M4 | M2 UI + 实机选项，逐 token 驱动实机 | 待做 |
| M5 | 加入对话功能 | 待做 |

## M3（已完成，回溯见 tag `M3`）

1. 7 权重 → factor 融合 → 下发（`face/face_driver.py`）
2. 实机联调（`face/server.py` + `webui/`）

部署包：`face_m3.zip`，说明见 `face/README_DEPLOY.md`。  
需 M3 可视化工具时：`git checkout M3`（含 `M3test/FKCN_visualize/`）。

## M4

**目标**：M2 Gradio UI 增加「实机」开关；工控机勾选后，ERNIE 逐 token 驱动 29 DOF。

**单通道 10fps**（屏幕与实机同步，不做 25fps 双通道）：
- `FaceFrameInterpolator` 改为 **10fps**，`MOVE_TIME_MS=100`
- 每帧：FKCN 7d 插值 → fuse 29 factor → Grimace 渲染 + `FaceDriver` 下发（同一帧节拍）
- 实机关闭时仍 10fps 预览（与实机一致，避免两套节奏）

**工控机部署**（全仓 + 仓库根目录单一 `venv`，与 M3 相同习惯）：
- **一个 venv**：在仓库根目录 `python -m venv venv && source venv/bin/activate`
- **一次安装**：`pip install -r requirements.txt`（已含 torch、gradio、pyserial 等；`face/` 仅代码，无单独环境）
- **模型与 scaler**：同步 `models/`（或工控机已有路径）、`outputs/scalers/`；路径以 `config.py` 为准
- **端口**：Gradio 7860；M4 只跑 `stream_ernie_vad_ui.py`，**不要同时起 M3 的 `face/server.py`（占串口）**
- **串口**：M4 进程内 import `FaceDriver`；配置见 `face/face_driver.py`（`SERIAL_PORT`、`MOVE_TIME_MS=100`）

启动：
```bash
source venv/bin/activate
python examples/stream_ernie_vad_ui.py
```
浏览器打开 Gradio，勾选「实机」。

## M5

加入对话功能。
