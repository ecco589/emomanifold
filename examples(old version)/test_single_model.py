#!/usr/bin/env python3
"""
单模型测试脚本示例
演示如何使用测试框架测试单个模型
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
    print("单模型测试示例")
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
    
    # 测试单个模型（ernie）
    model_name = "ernie"  # 可以改为 'glm' 或 'deepseek'
    
    # 检查模型是否可用
    model_config = config.ModelConfig(model_name)
    if not model_config.is_available():
        print(f"\n错误: 模型 {model_name} 不可用")
        print(f"模型路径: {model_config.local_path}")
        print(f"\n可用的模型: {config.ModelConfig.get_available_models()}")
        return 1
    
    # 测试模型，生成所有类型的图表
    success = framework.test_model(
        model_name=model_name,
        test_texts=test_texts,
        plot_types=['all']  # 生成所有类型的图表
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
