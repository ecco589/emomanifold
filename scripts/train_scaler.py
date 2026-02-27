"""
训练AdaptiveScaler脚本
使用训练语料库训练scale和bias参数
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.model_loader import ModelLoader, CausalModelLoader
from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.adaptive_scaler import AdaptiveScaler


def load_corpus(corpus_path=None):
    """加载训练语料库"""
    if corpus_path is None:
        corpus_path = project_root / "data" / "training_corpus.txt"
    
    corpus_path = Path(corpus_path)
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"语料库文件不存在: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    return corpus


def train_scaler(model_name='glm', corpus_path=None, output_path=None):
    """
    训练scaler
    
    Args:
        model_name: 模型名称 ('glm', 'deepseek', 'ernie')
        corpus_path: 语料库路径
        output_path: 输出路径
    """
    print(f"开始训练 {model_name.upper()} 模型的 AdaptiveScaler")
    print("=" * 60)
    
    # 1. 加载语料库
    print("\n[步骤1] 加载训练语料库...")
    corpus = load_corpus(corpus_path)
    print(f"  语料库大小: {len(corpus)} 句")
    
    # 2. 设置模型
    print(f"\n[步骤2] 设置模型: {model_name}")
    original_model = config.CURRENT_MODEL
    config.CURRENT_MODEL = model_name
    
    try:
        # 3. 加载模型
        # ERNIE 在 chat 场景使用 CausalLM，Scaler 需在同一架构上训练以匹配流式投影
        print(f"\n[步骤3] 加载 {model_name.upper()} 模型...")
        if model_name == "ernie":
            print("  使用 CausalLM（与 stream_ernie_vad 一致）")
            model_loader = CausalModelLoader(model_name)
        else:
            model_loader = ModelLoader()
        model_loader.load()
        
        # 4. 加载VAD词库
        print(f"\n[步骤4] 加载VAD词库...")
        dimension_extractor = DimensionExtractor(
            model_loader.get_output_embedding_matrix(),
            model_loader.tokenizer
        )
        vad_lexicon = dimension_extractor.load_vad_lexicon()
        
        # 5. 提取情感维度
        print(f"\n[步骤5] 提取情感维度...")
        dimensions = dimension_extractor.extract_all_dimensions(vad_lexicon)
        print(f"  成功提取 {len(dimensions)} 个情感维度: {list(dimensions.keys())}")
        
        # 6. 创建投影器（不集成scaler，用于训练）
        print(f"\n[步骤6] 创建情感投影器...")
        projector = EmotionProjector(dimensions)
        
        # 7. 创建scaler
        print(f"\n[步骤7] 创建AdaptiveScaler...")
        scaler = AdaptiveScaler(vad_lexicon)
        
        # 8. 训练scaler
        print(f"\n[步骤8] 开始训练...")
        scaler.train(corpus, model_loader, projector, verbose=True)
        
        # 9. 保存scaler
        if output_path is None:
            output_dir = project_root / "outputs" / "scalers"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model_name}_scaler.json"
        
        print(f"\n[步骤9] 保存scaler...")
        scaler.save(str(output_path))
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"Scaler已保存到: {output_path}")
        
        return scaler
        
    finally:
        # 恢复config
        config.CURRENT_MODEL = original_model


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练AdaptiveScaler')
    parser.add_argument('--model', type=str, default='glm',
                       choices=['glm', 'deepseek', 'ernie'],
                       help='模型名称 (默认: glm)')
    parser.add_argument('--corpus', type=str, default=None,
                       help='训练语料库路径 (默认: data/training_corpus.txt)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径 (默认: outputs/scalers/{model}_scaler.json)')
    
    args = parser.parse_args()
    
    train_scaler(
        model_name=args.model,
        corpus_path=args.corpus,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
