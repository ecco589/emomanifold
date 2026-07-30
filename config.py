"""
EmotionManifold 项目配置文件
"""
from pathlib import Path
from typing import Dict, Optional
import torch


# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 模型路径配置
MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # 模型下载缓存目录

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
VAD_LEXICON_DIR = DATA_DIR / "vad_lexicon"

# 输出配置
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 情感维度配置（默认 2D；qwen 使用 3D VAD）
EMOTION_DIMENSIONS = ["valence", "arousal"]
EMOTION_DIMENSIONS_3D = ["valence", "arousal", "dominance"]

# 流式/默认主模型（可通过环境变量 EMOTION_MODEL 覆盖）
DEFAULT_STREAM_MODEL = "qwen"


class ModelConfig:
    """模型配置管理类"""
    
    # 支持的模型配置
    MODELS = {
        "deepseek": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "local_path": MODEL_CACHE_DIR / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-1___5B",
            "is_bidirectional": False,
        },
        "ernie": {
            "model_name": "PaddlePaddle/ERNIE-4.5-0.3B-PT",
            # 优先用仓库内 models/（部署包自带），否则回退到 modelscope home 缓存
            "local_path": (
                MODEL_CACHE_DIR / "PaddlePaddle" / "ERNIE-4.5-0.3B-PT"
                if (MODEL_CACHE_DIR / "PaddlePaddle" / "ERNIE-4.5-0.3B-PT").exists()
                else Path.home() / ".cache" / "modelscope" / "hub" / "models" / "PaddlePaddle" / "ERNIE-4.5-0.3B-PT"
            ),
            "is_bidirectional": True,
            "emotion_dimensions": ["valence", "arousal"],
            "vad_lexicon": VAD_LEXICON_DIR / "vad_lexicon_chinese_ernie.json",
            "use_causal_lm": True,
        },
        "qwen": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": (
                MODEL_CACHE_DIR / "Qwen" / "Qwen2.5-0.5B-Instruct"
                if (MODEL_CACHE_DIR / "Qwen" / "Qwen2.5-0.5B-Instruct").exists()
                else (
                    MODEL_CACHE_DIR / "Qwen" / "Qwen2___5-0___5B-Instruct"
                    if (MODEL_CACHE_DIR / "Qwen" / "Qwen2___5-0___5B-Instruct").exists()
                    else Path.home()
                    / ".cache"
                    / "modelscope"
                    / "hub"
                    / "models"
                    / "Qwen"
                    / "Qwen2.5-0.5B-Instruct"
                )
            ),
            "is_bidirectional": False,
            "emotion_dimensions": ["valence", "arousal", "dominance"],
            "vad_lexicon": VAD_LEXICON_DIR / "vad_lexicon_chinese_qwen.json",
            "use_causal_lm": True,
        },
        "glm": {
            "model_name": "ZhipuAI/chatglm3-6b",
            "local_path": Path.home() / ".cache" / "modelscope" / "hub" / "models" / "ZhipuAI" / "chatglm3-6b",
            "is_bidirectional": False,
        },
    }
    
    def __init__(self, model_name: str = "glm"):
        """
        初始化模型配置
        
        Args:
            model_name: 模型名称 ('deepseek', 'ernie', 'qwen', 'glm')
        """
        if model_name not in self.MODELS:
            raise ValueError(f"不支持的模型: {model_name}，支持的模型: {list(self.MODELS.keys())}")
        
        self.model_name = model_name
        self.config = self.MODELS[model_name]
    
    @property
    def model_id(self) -> str:
        """获取模型ID"""
        return self.config["model_name"]
    
    @property
    def local_path(self) -> Path:
        """获取本地模型路径"""
        return self.config["local_path"]
    
    @property
    def is_bidirectional(self) -> bool:
        """是否为双向模型"""
        return self.config["is_bidirectional"]

    @property
    def emotion_dimensions(self) -> list:
        """该模型使用的情感维度列表"""
        return self.config.get("emotion_dimensions", EMOTION_DIMENSIONS)

    @property
    def vad_lexicon_path(self) -> Path:
        """模型对应的 VAD 词库路径"""
        return self.config.get("vad_lexicon", VAD_LEXICON_DIR / "vad_lexicon_chinese.json")

    @property
    def use_causal_lm(self) -> bool:
        """流式/chat 场景是否使用 CausalLM"""
        return self.config.get("use_causal_lm", not self.is_bidirectional)
    
    def is_available(self) -> bool:
        """检查模型是否可用（路径是否存在）"""
        return self.local_path.exists()
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取所有可用的模型列表"""
        return [name for name, config in cls.MODELS.items() 
                if config["local_path"].exists()]
    
    @classmethod
    def get_all_models(cls) -> list:
        """获取所有支持的模型列表"""
        return list(cls.MODELS.keys())


def get_active_model() -> str:
    import os
    return os.environ.get("EMOTION_MODEL", DEFAULT_STREAM_MODEL)


# 默认模型配置（向后兼容）
CURRENT_MODEL = get_active_model()
_default_config = ModelConfig(CURRENT_MODEL)
MODEL_NAME = _default_config.model_id
MODEL_LOCAL_PATH = _default_config.local_path

# 设备配置（M1 Mac 优先使用 MPS）
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
