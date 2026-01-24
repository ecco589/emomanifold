#!/usr/bin/env python3
"""
ERNIE模型交互式生成测试脚本
支持通过终端与ERNIE模型进行对话交互
"""
import sys
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def load_ernie_model():
    """加载ERNIE模型和分词器"""
    print("=" * 60)
    print("ERNIE模型交互式生成测试")
    print("=" * 60)
    
    # 获取模型配置
    model_config = config.ModelConfig("ernie")
    
    # 检查模型是否可用
    if not model_config.is_available():
        print(f"\n错误: ERNIE模型不可用")
        print(f"模型路径: {model_config.local_path}")
        print(f"\n请确保模型已下载到上述路径")
        return None, None
    
    model_path = str(model_config.local_path)
    print(f"\n使用本地模型路径: {model_path}")
    
    # 检测设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"使用设备: {device}")
    
    # 优先尝试使用ModelScope，如果失败则使用transformers
    use_modelscope = False
    try:
        from modelscope import AutoModelForCausalLM as MSCopeAutoModelForCausalLM
        from modelscope import AutoTokenizer as MSCopeAutoTokenizer
        use_modelscope = True
        print("\n使用ModelScope库加载模型...")
    except ImportError:
        print("\nModelScope不可用，使用transformers库...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("\n错误: 未找到transformers库")
            print("请运行: pip install transformers")
            return None, None
    
    try:
        print("加载分词器...")
        if use_modelscope:
            tokenizer = MSCopeAutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        print("✓ 分词器加载成功")
        
        print("加载模型（这可能需要一些时间）...")
        # 选择数据类型
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        
        if use_modelscope:
            model = MSCopeAutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
            )
            # 手动移动到设备（不使用device_map避免需要accelerate）
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
            )
            model = model.to(device)
        
        model.eval()  # 设置为评估模式
        print("✓ 模型加载成功！\n")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_text(model, tokenizer, prompt, max_new_tokens=1024):
    """
    生成文本
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 用户输入的提示词
        max_new_tokens: 最大生成token数
        
    Returns:
        生成的文本
    """
    try:
        # 准备模型输入
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板（如果支持）
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # 如果apply_chat_template失败，直接使用prompt
                text = prompt
        else:
            text = prompt
        
        # 分词并转换为tensor
        model_inputs = tokenizer(
            [text], 
            add_special_tokens=False, 
            return_tensors="pt"
        )
        
        # 获取模型设备
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
        
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 提取新生成的token IDs
        # model_inputs是字典，使用字典访问方式
        input_ids_length = len(model_inputs['input_ids'][0])
        output_ids = generated_ids[0][input_ids_length:].tolist()
        
        # 解码生成的文本
        generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return generate_text
        
    except Exception as e:
        return f"生成错误: {e}"


def interactive_chat(model, tokenizer):
    """交互式对话循环"""
    print("=" * 60)
    print("开始交互式对话")
    print("输入 'quit', 'exit' 或 'q' 退出")
    print("=" * 60)
    print()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 生成回复
            print("ERNIE: ", end="", flush=True)
            response = generate_text(model, tokenizer, user_input)
            print(response)
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，退出...")
            break
        except EOFError:
            print("\n\n检测到EOF，退出...")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ERNIE模型交互式生成测试")
    parser.add_argument(
        "--test",
        type=str,
        help="测试模式：直接生成指定prompt的回复（非交互式）",
        default=None
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="最大生成token数",
        default=1024
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_ernie_model()
    
    if model is None or tokenizer is None:
        return 1
    
    # 测试模式：直接生成并退出
    if args.test:
        print(f"\n测试模式: 生成回复")
        print(f"输入: {args.test}")
        print("=" * 60)
        response = generate_text(model, tokenizer, args.test, max_new_tokens=args.max_tokens)
        print(f"输出: {response}")
        print("=" * 60)
        return 0
    
    # 交互式模式
    try:
        interactive_chat(model, tokenizer)
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
