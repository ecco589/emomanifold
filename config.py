"""
EmotionManifold 项目配置文件
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 模型配置
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # 模型下载缓存目录

# 本地模型路径（ModelScope下载后的实际路径）
MODEL_LOCAL_PATH = MODEL_CACHE_DIR / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-1___5B"

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
VAD_LEXICON_DIR = DATA_DIR / "vad_lexicon"

# 设备配置（M1 Mac 优先使用 MPS）
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# 情感维度配置
EMOTION_DIMENSIONS = ["valence", "arousal"]

# 输出配置
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
