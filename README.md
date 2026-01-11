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
│   └── visualizer.py       # 可视化
├── data/                   # 数据目录
│   └── vad_lexicon/        # VAD词库
├── scripts/                # 脚本工具
├── examples/               # 示例代码
└── config.py               # 配置文件
```

## 环境要求

- Python 3.8+
- M1 MacBook Pro（已优化 MPS 支持）
- 依赖见 `requirements.txt`

## 安装步骤

1. 克隆项目（如果从 Git 获取）

2. **创建并激活虚拟环境**（推荐，避免全局污染）：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   ```
   详细说明请参考 [SETUP.md](SETUP.md)

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 下载模型：
   ```bash
   modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```
   或使用 SDK：
   ```python
   from modelscope import snapshot_download
   model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
   ```

## 使用说明

（待补充，MVP 完成后更新）

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
