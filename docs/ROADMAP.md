# EmotionManifold 分阶段开发路线图

> **终极目标**：30+ 自由度面部机器人，逐 token 展示表情。用户语音输入 → 机器人语音输出，每个 token 与面部表情对齐。本地部署 ERNIE 0.3B。

> **原则**：先完成最小功能闭环（MVP），再逐步叠加能力。Mac 模拟 → Linux 虚拟环境 → 实机部署。

---

## 一、整体里程碑概览

| 阶段 | 环境 | 核心功能 | 输入方式 | 输出方式 |
|------|------|----------|----------|----------|
| **M1** | Mac | 文字→ERNIE→逐 token 显示 VAD | 键盘输入 | 终端/图形显示 VAD |
| **M2** | Mac | + 表情模拟（Grimace 渲染） | 同上 | + 逐 token 表情图 |
| **M3** | Mac | + 按键录音 → ASR → 文字 | 按下录音、松开结束 | 同上 |
| **M4** | Mac | + TTS 语音输出 | 同上 | 语音 + VAD/表情 |
| **M5** | Mac | 时间对齐（TTS 时长 ↔ VAD/表情） | 同上 | 对齐的完整闭环 |
| **M6** | Linux | 迁移与适配 | 同上 | 同上 |
| **M7** | 机器人 | 实机 30 DOF 面部控制 | 同上 → 逐步加实时语音 | 实机表情驱动 |

---

## 二、阶段详解

### M1：文字输入 → ERNIE 流式输出 → 逐 token 显示 VAD（Mac MVP）

**目标**：验证「边生成边投影、边显示」的完整链路，无需语音。

**输入**：键盘输入文本，回车发送。

**输出**：
- 终端流式打印模型生成的每个 token
- 同步显示该 token 的 VAD 值（V, A 或 V+A）
- 可选：简单的实时曲线图（如终端绘图或 matplotlib 动画）

**技术要点**：
1. **流式生成**：使用 `model.generate(..., max_new_tokens=1)` 循环，或 HuggingFace `TextIteratorStreamer` 配合 generation 回调
2. **逐 token 隐藏状态**：每生成 1 个 token，用当前完整序列做一次前向传播，取**最后一个位置**的 hidden state 进行投影
3. **投影**：复用现有 `EmotionProjector.project_hidden_state()` + `AdaptiveScaler`

**交付物**：
- `examples/stream_ernie_vad.py`：流式聊天 + 逐 token 打印 VAD
- 可选：`examples/stream_ernie_vad_visual.py`：带简单实时曲线的版本

**验收**：输入一句话，能看到 token 一个个出来，每个 token 旁有 VAD 数值。

**M1 已完成** ✓ 终端版 `stream_ernie_vad.py`，Gradio UI `stream_ernie_vad_ui.py`（悬停 token 显示 VAD），CausalLM Scaler 校准。

**M2 已完成** ✓ 沿用 M1 的 `stream_ernie_vad.py` / `stream_ernie_vad_ui.py`，新增 `generate_with_face_stream`、`src/grimace_utils.py`、`src/grimace_renderer.py`；UI 右侧逐 token 表情图。

---

### M2：逐 token 表情模拟（Grimace 渲染）

**目标**：将 VAD 映射为可看见的表情，在 Mac 上模拟机器人「面部」。

**输入**：沿用 M1（键盘输入）。

**输出**：
- 流式文字 + VAD
- 每个新 token 对应一张表情图（或同一窗口内刷新），使用现有 FKCN + Grimace 流程

**技术要点**：
1. 每产生一个 token 的 VAD → FKCN 得到 7 维权重 → Grimace 渲染
2. 可先用「弹窗/刷新图片」方式，不必实时 30fps，以验证逻辑为主
3. 平滑：对上一步的 FKCN 权重做 EMA，避免表情抖动

**交付物**：
- 扩展 `stream_ernie_vad.py` 或新建 `stream_ernie_face.py`
- 可复用 `render_grimace_images.py` 中的逻辑，改为「单帧渲染 + 流式更新」

**验收**：生成一句话时，能逐 token 看到表情变化（例如从平静→开心→惊讶）。

---

### M3：按键录音 → ASR → 文字输入

**目标**：用「按键录音」替代键盘输入，为后续语音交互打基础。

**输入**：按住键录音，松开结束；录音结束后自动送 ASR。

**输出**：ASR 转写文字 → 送入 ERNIE（同 M1/M2）。

**技术要点**：
1. 录音：`sounddevice` 或 `pyaudio`，按键状态用 `keyboard` 或 `pynput`
2. ASR：优先选本地/轻量方案，如：
   - FunASR（阿里，支持离线）
   - Whisper（CPU/Small 可在 Mac 上跑）
   - 或先接在线 API 做原型（如阿里云、讯飞）

**交付物**：
- `examples/push_to_talk_asr.py`：独立按键录音 + ASR 测试
- 整合到主流程：`examples/stream_ernie_with_asr.py`（按键录音 → ASR → 流式 ERNIE + VAD）

**验收**：按住说话、松开后，能正确转写成文字并送给 ERNIE。

---

### M4：TTS 语音输出

**目标**：机器人「出声」，而不仅是文字和表情。

**输入**：沿用 M3（按键录音 → ASR）。

**输出**：
- 流式文字 + VAD + 表情（同 M2）
- 整句或流式 TTS 播放

**技术要点**：
1. TTS 选项：Edge-TTS（免费）、VITS、Coqui TTS、Bark 等；优先本地/离线
2. 先做「整句 TTS」：等整句生成完再合成播放，保证稳定
3. 后续可探索流式 TTS（如逐句或按 chunk）以降低延迟

**交付物**：
- `examples/stream_ernie_with_tts.py`：在 M3 基础上增加 TTS 播放

**验收**：说完一句话后，能听到机器人用语音回复。

---

### M5：时间对齐（TTS 时长 ↔ VAD/表情）

**目标**：把每个 token 的 VAD/表情与 TTS 的播放时间对齐。

**输入**：同 M4。

**输出**：
- 播放 TTS 时，根据时间轴驱动 VAD 曲线或表情
- 例如：TTS 返回每个字的起止时间（或 phoneme 级），将 token 投影到时间轴上，插值得到连续 VAD/表情轨迹

**技术要点**：
1. TTS 对齐信息：部分 TTS 支持返回字/音素级时间戳（如 MFA、Whisper alignment、VITS 的 duration 等）
2. Token–时间映射：若 TTS 以「整句」为单位，可用简单线性分配（按 token 数量平分时长）；若有字级对齐，则用字与 token 的对应关系
3. 播放时按时间轴采样 VAD/表情，驱动 Grimace 或后续的 30 DOF 模型

**交付物**：
- `src/alignment.py`：Token–时间戳对齐工具
- 整合到带 TTS 的流程中

**验收**：听语音时，表情变化与 spoken 内容大致同步。

---

### M6：迁移到 Linux 虚拟环境

**目标**：在 Linux 上复现 Mac 上的 MVP，为实机部署做准备。

**技术要点**：
1. 依赖：`requirements.txt`、系统级库（如 ASR/TTS 的底层）
2. 设备：CUDA/MPS → 在 Linux 上主要为 CUDA 或 CPU
3. 文档：`docs/LINUX_SETUP.md`，记录环境、模型路径、已知问题

**验收**：在 Linux 虚拟机/服务器上，完整跑通 M5 流程。

---

### M7：实机部署（30 DOF 面部机器人）

**目标**：用实机硬件替代 Grimace 模拟，驱动真实面部。

**技术要点**：
1. 30 DOF 模型：需要「VAD/表情权重 → 30 维关节角」的映射（可学习或规则）
2. 硬件接口：串口、ROS、自定义 SDK 等
3. 同步：TTS 播放与 30 DOF 指令的时间对齐，考虑通信延迟
4. 性能：在嵌入式/边缘设备上，ERNIE 0.3B 的推理速度需满足实时要求

**验收**：在实机上完成「按键录音 → ASR → ERNIE → TTS + 表情」的端到端演示。

---

## 三、推荐执行顺序（按周拆解）

| 周 | 里程碑 | 具体任务 |
|----|--------|----------|
| 1 | M1 | 实现流式 ERNIE 生成 + 逐 token 隐藏状态提取；写 `stream_ernie_vad.py`；终端打印 VAD |
| 2 | M2 | 整合 FKCN + Grimace，逐 token 渲染表情图；可加简单平滑 |
| 3 | M3 | 按键录音模块；接入 ASR；整合到主流程 |
| 4 | M4 | 接入 TTS；实现「说完话→听到回复」 |
| 5 | M5 | 设计并实现 Token–时间对齐；TTS 播放时驱动表情 |
| 6+ | M6/M7 | Linux 迁移；实机适配与联调 |

---

## 四、MVP 最小闭环定义（M1–M2）

**最小可演示功能**：
1. 键盘输入一句话
2. ERNIE 流式生成回复
3. 每个 token 实时显示 VAD 值
4. 每个 token 对应一张表情图（Grimace 渲染）

**不包含**：语音输入、语音输出、实机、时间对齐。

完成 M1–M2 后，即可在 Mac 上展示「文字对话 + 逐 token 表情」的完整逻辑，为后续加语音和对齐打好基础。

---

## 五、附录：技术选型备忘

| 模块 | 推荐方案（Mac MVP） | 备选 |
|------|---------------------|------|
| 流式生成 | `generate(max_new_tokens=1)` 循环 | `TextIteratorStreamer` |
| 录音 | `sounddevice` | `pyaudio` |
| 按键检测 | `pynput`（跨平台） | `keyboard`（需 root） |
| ASR | FunASR / Whisper-small | 在线 API（原型） |
| TTS | Edge-TTS / Coqui | 在线 API（原型） |
| 表情渲染 | Grimace（现有） | 30 DOF 模型（实机阶段） |
