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
3. **平滑处理**：通过滤波器（EMA）将跳动的情感分值转化为平滑轨迹
4. **物理映射**：将连续坐标映射到面部机器人的控制参数（后续实现）

## 项目结构

```
EmotionManifold/
├── src/                    # 核心代码模块
│   ├── model_loader.py     # 模型加载
│   ├── dimension_extractor.py  # 情感维度提取
│   ├── emotion_projector.py   # Token-level情感投影
│   ├── visualizer.py       # 可视化
│   └── test_framework.py   # 统一测试框架
├── data/                   # 数据目录
│   └── vad_lexicon/        # VAD词库
├── scripts/                # 脚本工具
│   ├── download_model.py   # 模型下载脚本
│   └── install_dependencies.sh  # 依赖安装脚本
├── examples/               # 示例代码
│   ├── test_unified.py     # 统一模型测试脚本（推荐）
│   └── test_single_model.py  # 单模型测试示例
├── config.py               # 配置文件（包含模型配置管理）
└── outputs/                # 输出目录（测试结果）
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

**测试所有可用模型**（推荐）：
```bash
python examples/test_unified.py
```

**测试单个模型**：
```bash
python examples/test_single_model.py
```
编辑脚本中的 `model_name` 变量选择模型（'glm', 'ernie', 'deepseek'）

**编程方式**：
```python
from src.test_framework import ModelTestFramework

framework = ModelTestFramework()
test_texts = ["我今天非常开心！"]
framework.test_model("glm", test_texts, plot_types=['trajectory'])
```

### 支持的模型

- **GLM** (ChatGLM3-6B): 单向模型
- **ERNIE** (ERNIE-4.5-0.3B-PT): 双向模型  
- **DeepSeek** (DeepSeek-R1-Distill-Qwen-1.5B): 单向模型

模型配置在 `config.py` 中的 `ModelConfig` 类中管理。

## 开发状态

- [x] 项目结构搭建
- [ ] 模型加载模块
- [ ] 情感维度提取
- [ ] Token-level 投影
- [ ] 可视化功能
- [ ] MVP 验证

## 参考资料

- DeepSeek-R1-Distill-Qwen-1.5B: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- VAD 情感模型相关研究
