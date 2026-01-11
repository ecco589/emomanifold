#!/usr/bin/env python3
"""
模型下载脚本
使用 ModelScope SDK 下载 DeepSeek-R1-Distill-Qwen-1.5B 模型
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modelscope import snapshot_download
import config

def download_model():
    """下载模型"""
    print(f"开始下载模型: {config.MODEL_NAME}")
    print(f"模型将保存到: {config.MODEL_CACHE_DIR}")
    
    try:
        model_dir = snapshot_download(
            config.MODEL_NAME,
            cache_dir=str(config.MODEL_CACHE_DIR)
        )
        print(f"\n模型下载成功")
        print(f"模型路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"\n模型下载失败: {e}")
        print("\n提示：如果下载失败，可以尝试使用命令行方式：")
        print(f"  modelscope download --model {config.MODEL_NAME}")
        return None

if __name__ == "__main__":
    download_model()
