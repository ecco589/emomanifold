#!/usr/bin/env python3
"""
测试Token显示优化
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_loader import ModelLoader
import config


def main():
    print("=" * 60)
    print("Token显示优化测试")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "我今天非常开心，心情特别好！",
        "听到这个消息，我感到非常失望和沮丧。",
        "这个结果让我既惊讶又兴奋。",
        "我现在很平静，感觉很放松。"
    ]
    
    try:
        # 加载模型
        print("\n加载模型...")
        model_loader = ModelLoader()
        model_loader.load()
        
        print("\n测试Token显示:")
        print("-" * 60)
        
        for idx, text in enumerate(test_texts):
            print(f"\n测试文本 {idx + 1}: {text}")
            
            # 获取token IDs和可读tokens
            token_ids, tokens = model_loader.tokenize(text, return_tokens=True)
            
            print(f"  Token数量: {len(tokens)}")
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {token_ids[:10]}..." if len(token_ids) > 10 else f"  Token IDs: {token_ids}")
            
            # 验证每个token是否可读
            readable_count = sum(1 for t in tokens if t and not t.startswith('['))
            print(f"  可读Token数: {readable_count}/{len(tokens)}")
        
        print("\n" + "=" * 60)
        print("Token显示测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
