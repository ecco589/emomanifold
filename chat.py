#!/usr/bin/env python3
"""
DeepSeek 模型交互式对话接口
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model_loader import ModelLoader


def main():
    print("=" * 60)
    print("DeepSeek 模型交互式对话")
    print("=" * 60)
    print("提示: 输入 'exit' 或 'quit' 退出对话\n")
    
    # 加载模型
    print("正在加载模型，请稍候...")
    try:
        loader = ModelLoader()
        loader.load()
        print("模型加载完成！\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 对话历史
    history = []
    
    # 交互循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                print("\n再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 生成回复
            print("模型: ", end="", flush=True)
            reasoning, response = loader.chat(user_input, history=history, return_reasoning=True)
            
            # 显示推理过程和实际输出
            if reasoning:
                print(f"\n[推理过程]\n{reasoning}\n")
                print("[实际输出] ", end="", flush=True)
            
            print(response)
            
            # 更新历史
            history.append((user_input, response))
            
            # 限制历史长度（避免内存过大）
            if len(history) > 10:
                history = history[-10:]
                
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    exit(main())
