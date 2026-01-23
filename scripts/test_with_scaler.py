"""
使用训练好的scaler测试脚本
对四句测试文本进行情感投影，并生成可视化结果
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.model_loader import ModelLoader
from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.adaptive_scaler import AdaptiveScaler
from src.visualizer import EmotionVisualizer


def main():
    """主函数"""
    print("=" * 60)
    print("使用训练好的Scaler测试ERNIE模型")
    print("=" * 60)
    
    # 测试文本（四句话）
    test_texts = [
        "我今天非常开心，心情特别好！",
        "听到这个消息，我感到非常失望和沮丧。",
        "这个结果让我既惊讶又兴奋。",
        "我现在很平静，感觉很放松。"
    ]
    
    # 设置模型
    model_name = 'ernie'
    original_model = config.CURRENT_MODEL
    config.CURRENT_MODEL = model_name
    
    try:
        # 步骤1: 加载模型
        print(f"\n[步骤1] 加载{model_name.upper()}模型...")
        model_loader = ModelLoader()
        model_loader.load()
        
        # 步骤2: 加载VAD词库
        print(f"\n[步骤2] 加载VAD词库...")
        dimension_extractor = DimensionExtractor(
            model_loader.get_output_embedding_matrix(),
            model_loader.tokenizer
        )
        vad_lexicon = dimension_extractor.load_vad_lexicon()
        
        # 步骤3: 提取情感维度
        print(f"\n[步骤3] 提取情感维度...")
        dimensions = dimension_extractor.extract_all_dimensions(vad_lexicon)
        print(f"成功提取 {len(dimensions)} 个情感维度: {list(dimensions.keys())}")
        
        # 步骤4: 加载训练好的scaler
        print(f"\n[步骤4] 加载训练好的AdaptiveScaler...")
        scaler_path = project_root / "outputs" / "scalers" / f"{model_name}_scaler.json"
        scaler = AdaptiveScaler(vad_lexicon)
        scaler.load(str(scaler_path))
        print(f"  Valence: scale={scaler.scales['valence']:.4f}, bias={scaler.biases['valence']:.4f}")
        print(f"  Arousal: scale={scaler.scales['arousal']:.4f}, bias={scaler.biases['arousal']:.4f}")
        
        # 步骤5: 创建投影器（集成scaler）
        print(f"\n[步骤5] 创建情感投影器（集成scaler）...")
        projector = EmotionProjector(dimensions, scaler=scaler)
        
        # 步骤6: 创建可视化器
        print(f"\n[步骤6] 创建可视化器...")
        output_dir = project_root / "outputs" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizer = EmotionVisualizer(output_dir=output_dir)
        
        # 步骤7: 对测试文本进行情感投影
        print(f"\n[步骤7] 对测试文本进行情感投影...")
        print("-" * 60)
        
        for idx, text in enumerate(test_texts):
            print(f"\n测试文本 {idx+1}: {text}")
            
            # 投影（应用scaler）
            emotion_coords, tokens = projector.project_text(model_loader, text, apply_scaling=True)
            
            # 打印结果
            print(f"  Token数量: {len(tokens)}")
            
            # 打印情感坐标统计
            for dim_name, values in emotion_coords.items():
                mean_val = sum(values) / len(values) if values else 0
                max_val = max(values) if values else 0
                min_val = min(values) if values else 0
                print(f"  {dim_name.capitalize()}: 均值={mean_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
            
            # 生成可视化
            save_prefix = output_dir / f"{model_name}_test_{idx+1}"
            visualizer.plot_all(
                emotion_coords,
                tokens=tokens,
                text=text,
                save_prefix=str(save_prefix)
            )
            print(f"  可视化图表已保存到: {save_prefix}_*.png")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print(f"所有结果已保存到: {output_dir}")
        print("=" * 60)
        
    finally:
        # 恢复config
        config.CURRENT_MODEL = original_model


if __name__ == "__main__":
    main()
