#!/usr/bin/env python3
"""
统一模型测试脚本
对所有模型运行相同的测试文本，生成trajectory图
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.test_framework import ModelTestFramework
import config


def main():
    """主函数"""
    print("=" * 60)
    print("统一模型测试 - 生成Trajectory图")
    print("=" * 60)
    
    # 测试文本（四句话）
    test_texts = [
        "我今天非常开心，心情特别好！",
        "听到这个消息，我感到非常失望和沮丧。",
        "这个结果让我既惊讶又兴奋。",
        "我现在很平静，感觉很放松。"
    ]
    
    # 创建测试框架
    framework = ModelTestFramework()
    
    # 测试所有可用模型，只生成trajectory图
    results = framework.test_all_models(
        test_texts=test_texts,
        output_subdir="unified",
        plot_types=['trajectory'],  # 只生成trajectory图
        clean_output=True
    )
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main())
