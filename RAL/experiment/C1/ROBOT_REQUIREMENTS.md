# C1 工控机端实现需求

> **给实机 Cursor 的实现说明。** 本文件描述工控机侧录制程序需求；分析在本机完成，见 `README.md`。

部署根目录：`/home/code/lyg/m4/EmotionManifold`  
C1 目录：`RAL/experiment/C1/`（与部署根并列）  
虚拟环境：`/home/code/venv`

---

## 目标

实现 **`RAL/experiment/C1/code/run_c1_capture.py`**（名称可调整），完成：

1. 读取 `data/stimuli.csv`（或 `data/stimuli_one.csv` 单条试跑）
2. 对每条用户句：Qwen 流式对话 + 实机 29 DOF 表情 + USB 外置相机录像
3. 导出 ideal VA（逐 token）+ 视频到固定目录
4. 终端/UI 显示进度（如 `3 / 20`）

**不要**同时运行 `stream_ernie_vad_ui.py` 或 `face/server.py`。

---

## 功能需求

### F1. USB 相机预览

- 检测 USB 外置相机（如 `/dev/video0`，可 `--camera` 参数覆盖）
- 启动后显示**实时预览窗口**（OpenCV 等），便于对准机器人脸
- 预览与录制共用同一路视频流
- 建议分辨率 **1080p、30fps**（与 Mirror 分析一致）

### F2. 逐条自动录制

对 `stimuli.csv` 每一行（列：`id,category,user_prompt,notes`）：

| 阶段 | 行为 |
|------|------|
| 开始 | **自动**开始 ffmpeg/OpenCV 写 `video.mp4` |
| 同期 | 调用现有管线：`generate_with_face_stream` + `RobotMotionSmoother` 驱动实机 |
| 结束 | 最后 token 完成后 **固定延迟 ~0.5s**（舵机到位）→ **自动**停录 |
| 间隔 | trial 之间可选 2–3s 中性复位 |

**时间对齐**：`source_va.csv` 中 `timestamp_s` 以**本条开始录制时刻**为 `t=0`（与视频第一帧对齐）。

### F3. 进度显示

终端或简单 GUI 必须显示：

```text
[C1] Trial 3 / 20  id=003  category=happy
     user: 今天中了彩票，太开心了！
     status: recording | generating | saving
```

### F4. 输出目录（固定）

```text
RAL/experiment/C1/result/recordings/
├── 001/
│   ├── video.mp4          # H.264, 1080p30, yuv420p
│   ├── source_va.csv      # 见下方格式
│   └── session.json       # 元数据
├── 002/
│   └── ...
```

**`source_va.csv` 格式**（Mirror 已支持）：

```csv
timestamp_s,valence,arousal,token
0.000,0.12,0.34,你
0.045,0.15,0.36,好
```

**`session.json` 建议字段**：`trial_id`, `category`, `user_prompt`, `assistant_text`, `duration_s`, `n_tokens`, `video_fps`, `camera_device`

### F5. 复用现有部署代码

- 模型：`load_chat_model_and_projector("qwen")`（`config.py` 默认 qwen，VA 二维）
- 实机：`RobotFaceSink` + `RobotMotionSmoother`（与 `stream_ernie_vad_ui.py` 相同逻辑）
- FKCN：`load_fkcn()`
- 系统提示词：与 UI 一致（简体中文、简洁回复）

### F6. 命令行接口

```bash
cd /home/code/lyg/m4/EmotionManifold
source /home/code/venv/bin/activate

# 单条试跑（先跑通流程）
python RAL/experiment/C1/code/run_c1_capture.py \
  --stimuli RAL/experiment/C1/data/stimuli_one.csv \
  --camera /dev/video0

# 正式 20 条
python RAL/experiment/C1/code/run_c1_capture.py \
  --stimuli RAL/experiment/C1/data/stimuli.csv \
  --camera /dev/video0
```

可选参数：`--out-dir`、`--start-id`、`--limit`、`--preview-only`（只开预览不测）

---

## 试跑流程（1 条）

1. `sudo chmod 777 /dev/usb_face`
2. 确认相机：`v4l2-ctl --list-devices`
3. 运行 `stimuli_one.csv`（仅 1 条）
4. 检查 `result/recordings/001/` 含 `video.mp4` + `source_va.csv`
5. 将整个 `result/recordings/` 拷到本机做 Mirror 分析

---

## 交付物

工控机 Cursor 只需提交：

- `RAL/experiment/C1/code/run_c1_capture.py`（及必要小模块）
- 若改依赖，在部署 `requirements.txt` 注明（如 `opencv-python`）

本仓库 C1 **不含**工控机录制代码；分析脚本保留在本机。

---

## 非目标

- 不在工控机安装 Py-Feat / Mirror
- 不需要 Gradio UI
- 不需要手机录制流程
