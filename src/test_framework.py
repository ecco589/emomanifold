"""
统一的模型测试框架
提供可复用的测试逻辑，避免代码重复
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config


class ModelTestFramework:
    """统一的模型测试框架"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化测试框架
        
        Args:
            output_dir: 输出目录，默认使用 config.OUTPUT_DIR
        """
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self, model_name: str):
        """
        设置并初始化模型相关的组件
        
        Args:
            model_name: 模型名称 ('glm', 'ernie', 'deepseek')
            
        Returns:
            Tuple: (model_loader, dimension_extractor, projector, visualizer)
        """
        # 延迟导入，避免循环依赖
        from src.model_loader import ModelLoader
        from src.dimension_extractor import DimensionExtractor
        from src.emotion_projector import EmotionProjector
        from src.visualizer import EmotionVisualizer
        
        # 创建模型配置
        model_config = config.ModelConfig(model_name)
        
        # 检查模型是否可用
        if not model_config.is_available():
            raise FileNotFoundError(f"模型路径不存在: {model_config.local_path}")
        
        # 临时设置config（为了兼容ModelLoader）
        original_model = config.CURRENT_MODEL
        config.CURRENT_MODEL = model_name
        config.MODEL_NAME = model_config.model_id
        config.MODEL_LOCAL_PATH = model_config.local_path
        
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
            
            # 步骤4: 创建投影器
            print(f"\n[步骤4] 创建情感投影器...")
            projector = EmotionProjector(dimensions)
            
            # 步骤5: 创建可视化器
            output_dir = self.output_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            visualizer = EmotionVisualizer(output_dir=output_dir)
            
            return model_loader, dimension_extractor, projector, visualizer
            
        finally:
            # 恢复config
            config.CURRENT_MODEL = original_model
    
    def test_text(self, projector, model_loader, 
                   text: str, visualizer=None,
                   save_prefix: Optional[str] = None, plot_types: List[str] = None) -> Dict:
        """
        测试单个文本的情感投影
        
        Args:
            projector: 情感投影器
            model_loader: 模型加载器
            text: 测试文本
            visualizer: 可视化器（可选）
            save_prefix: 保存文件前缀（可选）
            plot_types: 要生成的图表类型列表，可选值: ['trajectory', '2d', 'timeline', 'all']
            
        Returns:
            Dict: 包含emotion_coords和tokens的字典
        """
        print(f"\n测试文本: {text}")
        
        # 投影
        emotion_coords, tokens = projector.project_text(model_loader, text)
        
        # 打印结果
        print(f"  Token数量: {len(tokens)}")
        print(f"  Tokens: {tokens}")
        
        # 打印情感坐标统计
        for dim_name, values in emotion_coords.items():
            mean_val = sum(values) / len(values) if values else 0
            max_val = max(values) if values else 0
            min_val = min(values) if values else 0
            print(f"  {dim_name.capitalize()}: 均值={mean_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
        
        # 可视化
        if visualizer and save_prefix:
            plot_types = plot_types or ['trajectory']
            
            if 'all' in plot_types:
                visualizer.plot_all(
                    emotion_coords,
                    tokens=tokens,
                    text=text,
                    save_prefix=str(save_prefix)
                )
            else:
                if 'trajectory' in plot_types:
                    visualizer.plot_emotion_trajectory(
                        emotion_coords,
                        tokens=tokens,
                        title=f'Emotion Trajectories - {text}',
                        save_path=f"{save_prefix}_trajectory.png"
                    )
                
                if '2d' in plot_types and 'valence' in emotion_coords and 'arousal' in emotion_coords:
                    visualizer.plot_valence_arousal_2d(
                        emotion_coords['valence'],
                        emotion_coords['arousal'],
                        tokens=tokens,
                        title=f'Valence-Arousal 2D - {text}',
                        save_path=f"{save_prefix}_2d.png"
                    )
                
                if 'timeline' in plot_types:
                    visualizer.plot_timeline(
                        emotion_coords,
                        tokens=tokens,
                        title=f'Emotion Timeline - {text}',
                        save_path=f"{save_prefix}_timeline.png"
                    )
            
            print(f"  可视化图表已保存到: {save_prefix}_*.png")
        
        return {
            'emotion_coords': emotion_coords,
            'tokens': tokens,
            'text': text
        }
    
    def test_model(self, model_name: str, test_texts: List[str], 
                   output_subdir: Optional[str] = None,
                   plot_types: List[str] = None) -> bool:
        """
        测试指定模型
        
        Args:
            model_name: 模型名称 ('glm', 'ernie', 'deepseek')
            test_texts: 测试文本列表
            output_subdir: 输出子目录（可选，默认使用model_name）
            plot_types: 要生成的图表类型列表
            
        Returns:
            bool: 测试是否成功
        """
        print("=" * 60)
        print(f"测试模型: {model_name.upper()}")
        print("=" * 60)
        
        try:
            # 设置模型
            model_loader, dimension_extractor, projector, visualizer = self.setup_model(model_name)
            
            # 设置输出目录
            output_subdir = output_subdir or model_name
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            visualizer.output_dir = output_dir
            
            # 对测试文本进行情感投影
            print(f"\n[步骤5] 对测试文本进行情感投影...")
            print("-" * 60)
            
            for idx, text in enumerate(test_texts):
                save_prefix = output_dir / f"{model_name}_test_{idx+1}"
                self.test_text(
                    projector, model_loader, text, visualizer,
                    save_prefix=str(save_prefix),
                    plot_types=plot_types
                )
            
            print("\n" + "=" * 60)
            print(f"{model_name.upper()}模型测试完成")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n{model_name.upper()}模型测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_all_models(self, test_texts: List[str], 
                       models: Optional[List[str]] = None,
                       output_subdir: str = "unified",
                       plot_types: List[str] = None,
                       clean_output: bool = True) -> Dict[str, bool]:
        """
        测试所有可用模型
        
        Args:
            test_texts: 测试文本列表
            models: 要测试的模型列表（可选，默认测试所有可用模型）
            output_subdir: 输出子目录
            plot_types: 要生成的图表类型列表
            clean_output: 是否清理旧的输出文件
            
        Returns:
            Dict: 模型名称到测试结果的映射
        """
        print("=" * 60)
        print("统一模型测试")
        print("=" * 60)
        
        # 确定要测试的模型
        if models is None:
            models = config.ModelConfig.get_available_models()
        
        if not models:
            print("\n错误: 没有找到任何可用的模型")
            return {}
        
        print(f"\n将测试以下模型: {', '.join(models)}")
        
        # 清理旧文件
        if clean_output:
            output_base_dir = self.output_dir / output_subdir
            if output_base_dir.exists():
                print(f"\n清理旧的输出文件...")
                for subdir in output_base_dir.iterdir():
                    if subdir.is_dir():
                        shutil.rmtree(subdir)
                        print(f"  已删除: {subdir.name}/")
        
        # 逐个测试模型
        results = {}
        for model_name in models:
            success = self.test_model(
                model_name, test_texts,
                output_subdir=f"{output_subdir}/{model_name}",
                plot_types=plot_types
            )
            results[model_name] = success
            print("\n")
        
        # 总结
        print("=" * 60)
        print("测试总结")
        print("=" * 60)
        for model_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{model_name.upper()}: {status}")
        
        output_path = self.output_dir / output_subdir
        print(f"\n所有结果已保存到: {output_path}")
        
        return results
