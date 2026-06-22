# EmotionManifold

## 项目简介

EmotionManifold 是一个基于大语言模型（LLM）的连续多维情感提取系统。通过从模型的解嵌入矩阵中提取情感维度，实现 **Token-level** 的情感表达，使机器人面部动作更加自然、细腻。

## 核心思想

- **连续情感空间**：打破传统离散情感标签的局限，提取连续的多维情感坐标（如 VAD 模型）
- **Token-level 情感**：逐词提取情感，而非整句话匹配单一表情
- **无需训练**：通过向量算术在模型的解嵌入矩阵中找到情感维度的基轴向量

## 技术架构

1. **维度挖掘**：通过对比情感词簇的向量差值，定义 Valence（效价）和 Arousal（唤醒度）等情感轴
2. **实时投影**：将每个 Token 的隐藏状态投影到预定义的情感轴上
3. **Scaler 标定**：使用 `AdaptiveScaler` 在训练语料上拟合 scale/bias，使情感分值落在合理范围
4. **平滑处理**：通过滤波器（EMA）将跳动的情感分值转化为平滑轨迹
5. **表情映射**：通过 FKCN 将 VAD 坐标映射到 7 种基本表情的融合权重（neutral, happiness, surprise, fear, sadness, disgust, anger）
6. **渲染输出**：结合 grimace-main 将情感权重渲染为面部表情图像序列

## 项目结构

```
EmotionManifold/
├── src/                        # 核心代码模块
│   ├── model_loader.py         # 模型加载
│   ├── dimension_extractor.py  # 情感维度提取
│   ├── emotion_projector.py    # Token-level 情感投影
│   ├── adaptive_scaler.py      # 自适应缩放器（VAD 标定）
│   ├── fkcn.py                 # FKCN 模糊聚类，将 VAD 映射到 7 种基本表情权重
│   └── visualizer.py           # 情感轨迹可视化
├── data/                       # 数据目录
│   ├── vad_lexicon/            # VAD 词库
│   └── training_corpus.txt     # Scaler 训练语料
├── grimace-main/               # 表情渲染子项目（基于 Grimace 引擎）
│   └── ...                     # 将情感权重渲染为面部表情图像
├── scripts/                    # 脚本工具
│   ├── download_model.py       # 模型下载脚本
│   ├── install_dependencies.sh # 依赖安装脚本
│   ├── train_scaler.py         # 训练 AdaptiveScaler
│   ├── test_with_scaler.py     # 使用 Scaler 测试情感投影（推荐）
│   ├── generate_training_corpus.py  # 生成训练语料
│   └── render_grimace_images.py    # 将情感投影渲染为表情序列图
├── examples/                   # 示例代码
│   └── stream_ernie_vad_ui.py  # Gradio UI（流式 VAD + 表情）
├── config.py                   # 配置文件（模型配置、路径、设备）
├── outputs/                    # 输出目录（Scaler、图表、渲染结果）
└── logs/                       # 日志目录
```

## 环境要求

- Python 3.8+
- M1 MacBook Pro（已优化 MPS 支持）
- 依赖见 `requirements.txt`

## 快速开始

### 1. 环境设置

创建并激活虚拟环境（推荐，避免依赖冲突）：
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

安装依赖：
```bash
pip install -r requirements.txt
```

或使用项目脚本：
```bash
bash scripts/install_dependencies.sh
```

### 2. 下载模型

使用 ModelScope CLI：
```bash
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

或使用 Python SDK：
```python
from modelscope import snapshot_download
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
```

支持的模型会自动检测，详见 `config.py` 中的 `ModelConfig` 类。

### 3. 运行测试

**使用训练好的 Scaler 测试情感投影**（推荐）：
```bash
python scripts/test_with_scaler.py
```
该脚本会加载 ERNIE 模型与 VAD 维度，对四句测试文本做情感投影，并生成可视化图表到 `outputs/ernie/`。

**训练 Scaler**（需先准备 `data/training_corpus.txt`）：
```bash
python scripts/train_scaler.py
```
训练完成后，Scaler 保存至 `outputs/scalers/{model_name}_scaler.json`。

**流式聊天 + VAD + 表情（Gradio UI）**：双击 `启动ERNIE+VAD聊天.command` 或执行
```bash
source venv/bin/activate && python examples/stream_ernie_vad_ui.py
```
启动后打开浏览器访问 http://127.0.0.1:7860，悬停 token 显示 VAD。

**渲染表情图像**（需配置 grimace-main）：
```bash
python scripts/render_grimace_images.py
```

### 支持的模型

| 模型 | 配置名 | 类型 |
|------|--------|------|
| ChatGLM3-6B | `glm` | 单向 |
| ERNIE-4.5-0.3B-PT | `ernie` | 双向 |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek` | 单向 |

模型配置在 `config.py` 的 `ModelConfig` 类中管理，可通过 `config.CURRENT_MODEL` 切换。

## 版本管理

本地 Git 仓库，无远程。两条分支：

| 分支 | 说明 | 当前里程碑 |
|------|------|------------|
| `main` | 工程 demo：M1/M2 仿真 + M3 实机 | **M3**（tag `M3`） |
| `finalV1` | NeurIPS 2026 论文：VAD 读回、EMA 动力学、残差注入、B1–B5 实验 | **FinalNIPS**（tag `FinalNIPS`） |

`finalV1` 在 `main` 之上继续开发（含 M1/M2 全部历史）。

**主要 tag**：`v0.1.0` → `v3.0`（早期探索）→ `M1`（流式 VAD）→ `M2`（逐 token Grimace）→ **`M3`（VA→实机 29 DOF）** → `FinalNIPS`（论文终稿，finalV1 分支）

```bash
git checkout main      # demo 开发（当前默认）
git checkout finalV1   # 论文与实验
git checkout M2        # 钉在 M2 快照
git checkout M3        # 钉在 M3 快照（VA→实机）
git checkout FinalNIPS # 钉在论文终稿
```

M4–M5 见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 开发路线图

**M1 ✓** 逐 token VAD（`examples/stream_ernie_vad_ui.py`）

**M2 ✓** + Grimace 逐 token 表情

**M3 ✓** VA → FKCN → 29 DOF 实机（`face/`、`face_m3.zip`）

**M4** 逐 token 实机表情 · **M5** 对话功能

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 开发状态

- [x] 项目结构搭建
- [x] 模型加载模块（支持 GLM、ERNIE、DeepSeek）
- [x] 情感维度提取（VAD 词库）
- [x] Token-level 投影
- [x] AdaptiveScaler 标定
- [x] 可视化功能
- [x] FKCN 表情权重映射
- [x] Grimace 表情图像渲染
- [x] M1 流式 ERNIE + VAD（终端 + Gradio UI，token 悬停 tooltip）
- [x] M3 VA→实机：FKCN 融合 7 套表情 → 29 DOF（`face/`，工控机 Web 控制）

## 参考资料

- DeepSeek-R1-Distill-Qwen-1.5B: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- VAD 情感模型相关研究
