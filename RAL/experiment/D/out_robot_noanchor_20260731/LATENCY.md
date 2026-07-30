# Latency Metrics（本场 C1 延迟口径说明）

本文档随 `robot_handoff/out/` 一并打包，解释延迟表如何定义，避免与「整段文本再 project_text」或「含完整自回归生成」混淆。

## 方法差异（为何不能塞进同一格子）

| 系统 | 脸控更新方式 |
|------|----------------|
| **ours（本文）** | **每个生成 token** 更新一次表情（流式） |
| Mishra / Xpress | **整句回复结束后** 再分类，一次定表情（话轮级） |

因此延迟必须分清两层：**控制通路（per-token）** 与 **首次语义脸更新（TTFF）**。

---

## 指标定义

### A. ours 主指标：`ours_per_token`

对每个已生成 token，在 **LM 前向已得到该 token hidden 之后** 计时：

`project_hidden_state → FKCN → EMA smooth`

- **不含** 自回归 LM 前向（生成成本与出字共享，不算进「表情通路」）
- 每条刺激报告这些 per-token 耗时的 **均值**（ms）
- 这才对应本文方法：「token → 表情权重」

### B. ours 辅指标：`ours_ttff`（Time-to-First-Face-Update）

从 **开始生成回复** 到 **第一个内容 token 的控制权重就绪** 的墙钟时间。

- **包含** 第一个 token 的 LM 步进（否则脸上不可能更新）
- 用于描述「多快出现第一次与内容相关的表情」

### C. 基线话轮分类：`mishra_gpt` / `mishra_qwen` / `xpress_gpt` / `xpress_qwen`

在 **同一段 assistant 回复文本已就绪** 后：

- GPT 或本地 Qwen 完成分类（+ 映成权重）的耗时
- **仅分类**，不含生成回复

基线 **没有** 对等的 per-token 表情通路，故不报 `*_per_token`。

### D. 并排可比的 TTFF：`*_ttff`（基线）

`baseline_ttff = gen_wall_ms + classify_ms`

即：必须等整句生成完，再分类完，才第一次按语义改脸。

与 `ours_ttff` 同叙事：「从开始出回复 → 第一次语义相关脸控」。

---

## TTFF 主导因素（解读表时必读）

绝对毫秒数会随硬件与部署强烈变化；**方法对比的叙事是更新粒度（流式 vs 话轮），不是某一台机器上的绝对速度排行。** 各分量谁最大，见下。

### `ours_ttff` ≈ 首 token LM 前向（≫ 控制通路）

| 分量 | 典型量级（本场） | 主要取决于 |
|------|------------------|------------|
| **首 token LM forward**（含 prompt 预填 / 首步 decode） | **占 TTFF 绝大部分**（本场约数百 ms–数秒） | **CPU vs GPU、模型大小、精度、batch/KV cache、上下文长度** |
| `project_hidden → FKCN → EMA` | ~0.5–5 ms | CPU 上的小矩阵运算；相对 LM 可忽略 |

因此：`ours_ttff` 随推理设备几乎线性变化——同一代码在 GPU 上通常明显缩短，在弱 CPU 上会拉长。**换机复现时，优先对齐 `device`（cpu/cuda）与模型权重，而不是对齐绝对 ms。**

### 基线 `*_ttff` = `gen_wall` + `classify`

| 分量 | 主要取决于 |
|------|------------|
| **`gen_wall_ms`**（整句自回归） | 与 ours 同一本地 LM：**CPU/GPU**、回复 token 数、采样设置。token 越多，墙钟越长，基线 TTFF 被抬得越高 |
| **`mishra_gpt` / `xpress_gpt`（classify）** | **网络 RTT、代理、API 侧排队与模型**；与本机 CPU/GPU 关系弱 |
| **`mishra_qwen` / `xpress_qwen`（classify）** | 再次跑本地 LM：**同一 CPU/GPU 瓶颈**；本场 Qwen 分类有时远慢于 GPT API |

基线第一次语义改脸必须等「说完 + 分完类」，故 TTFF 通常由 **`gen_wall`（硬件绑定）+ 分类（API 或二次本地推理）** 共同主导；ours 只需付 **一次首 token 推理**。

### 本场采集机器型号（详见 `latency/machine_env.txt`）

| 项 | 本场实测 |
|----|----------|
| **CPU** | **Intel(R) N100**（4 核 / 4 线程，基频约 0.7–3.4 GHz） |
| **内存** | 约 16 GB |
| **独立 GPU** | 无 NVIDIA；仅核显 `Intel Corporation Device 46d1` |
| **推理设备** | `device_arg: cpu`（本地生成与 Qwen 分类均在 CPU） |
| **OS / 运行时** | Linux 5.15 x86_64；Python 3.10；torch 2.12.1+cu130 |

因此本场 `ours_ttff` / `gen_wall` / `*_qwen` 偏大属预期（弱 CPU、无 CUDA 推理）。GPT 分类延迟随代理与远端负载波动；连接失败时脚本会暂停重试代理（见文末运维）。

**写论文 / 复现时建议同时报告：** 上表 CPU/GPU、`device_arg`、模型 id、以及 `machine_env.txt`，避免把本场 CPU 上的绝对 TTFF 当成方法上限。

---

## 示意例子（同一条回复，20 token）

| 指标 | 含义 | 量级（示意） |
|------|------|----------------|
| `ours_per_token` | 每个 token 投影+FKCN | ~1–5 ms（几乎不受 GPU/CPU 主导） |
| `ours_ttff` | 第 1 个 token 可控 | ~数百 ms–数秒（**几乎全是首 token LM，跟设备走**） |
| `mishra_gpt` | 整句 GPT 分类 | ~1–3 s（网络/API） |
| `mishra_gpt_ttff` | 生成整句 + GPT 分类 | ~`gen_wall`（设备） + 1–3 s |

审稿人可读结论：本文是 **流式每 token 可更新**；基线是 **话轮结束后才更新**。主对比讲清粒度，而不是假装同一个「延迟」标量；绝对 TTFF 数字需结合硬件解读。

---

## 文件

- `latency/latency_raw.csv`：逐条原始值（`condition` 见上）
- `latency/latency_table.csv`：按 condition 汇总 mean±std
- `latency/machine_env.txt`：机器环境（平台、`device_arg`、torch 等）
- `c1/_latency_labels/{id}_ours_ctrl.json`：该条 per-token 控制耗时序列等

## 人脸录像

延迟口径 **不影响** 四路人脸录像逻辑（ours / mishra_gpt / xpress_gpt / random）。C2 不记录延迟。

## 运维

GPT 出现连接/地区限制等错误时，采集会 **暂停并重试代理**，恢复后再继续，避免半截坏结果。
