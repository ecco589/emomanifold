#!/usr/bin/env python3
"""
EmotionManifold MVP 测试脚本
验证核心功能：从模型输出层权重中提取情感维度，并进行Token-level投影
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_loader import ModelLoader
from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.visualizer import EmotionVisualizer
import config


def main():
    print("=" * 60)
    print("EmotionManifold MVP 测试")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "我今天非常开心，心情特别好！",
        "听到这个消息，我感到非常失望和沮丧。",
        "这个结果让我既惊讶又兴奋。",
        "我现在很平静，感觉很放松。"
    ]
    
    try:
        # 步骤1: 加载模型
        print("\n[步骤1] 加载模型...")
        model_loader = ModelLoader()
        model_loader.load()
        
        # 步骤2: 加载VAD词库
        print("\n[步骤2] 加载VAD词库...")
        dimension_extractor = DimensionExtractor(
            model_loader.get_output_embedding_matrix(),
            model_loader.tokenizer
        )
        vad_lexicon = dimension_extractor.load_vad_lexicon()
        
        # 步骤3: 提取情感维度
        print("\n[步骤3] 提取情感维度...")
        dimensions = dimension_extractor.extract_all_dimensions(vad_lexicon)
        print(f"成功提取 {len(dimensions)} 个情感维度: {list(dimensions.keys())}")
        
        # 步骤4: 创建投影器
        print("\n[步骤4] 创建情感投影器...")
        projector = EmotionProjector(dimensions)
        
        # 步骤5: 创建可视化器
        print("\n[步骤5] 创建可视化器...")
        visualizer = EmotionVisualizer()
        
        # 步骤6: 对测试文本进行情感投影
        print("\n[步骤6] 对测试文本进行情感投影...")
        print("-" * 60)
        
        for idx, text in enumerate(test_texts):
            print(f"\n测试文本 {idx + 1}: {text}")
            
            # 投影
            emotion_coords, tokens = projector.project_text(model_loader, text)
            
            # 打印结果
            print(f"  Token数量: {len(tokens)}")
            print(f"  Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"  Tokens: {tokens}")
            
            # 打印情感坐标统计
            for dim_name, values in emotion_coords.items():
                mean_val = sum(values) / len(values) if values else 0
                max_val = max(values) if values else 0
                min_val = min(values) if values else 0
                print(f"  {dim_name.capitalize()}: 均值={mean_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
            
            # 可视化
            save_prefix = config.OUTPUT_DIR / f"test_{idx+1}"
            visualizer.plot_all(
                emotion_coords,
                tokens=tokens,
                text=text,
                save_prefix=str(save_prefix)
            )
            print(f"  可视化图表已保存到: {save_prefix}_*.png")
        
        print("\n" + "=" * 60)
        print("MVP测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
