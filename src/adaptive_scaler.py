"""
自适应缩放器模块
通过训练语料库学习scale和bias参数，将投影值对齐到词典范围
"""
import torch
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import config


class AdaptiveScaler:
    """自适应缩放器，用于训练和保存scale/bias参数"""
    
    def __init__(self, vad_lexicon: Dict, device=None):
        """
        初始化缩放器
        
        Args:
            vad_lexicon: VAD词典字典
            device: 设备类型
        """
        self.device = device or config.DEVICE
        self.vad_lexicon = vad_lexicon
        
        # 将词典值映射到-1~1范围（在所有步骤前完成）
        self.mapped_lexicon = self._map_lexicon_to_range(vad_lexicon)
        
        # 获取词典词列表
        self.lexicon_words = list(self.mapped_lexicon.keys())
        
        # 为每个维度维护累积统计量
        self.stats = {
            'valence': {
                'sum_raw': 0.0,
                'sum_target': 0.0,
                'sum_raw_squared': 0.0,
                'sum_raw_target': 0.0,
                'count': 0
            },
            'arousal': {
                'sum_raw': 0.0,
                'sum_target': 0.0,
                'sum_raw_squared': 0.0,
                'sum_raw_target': 0.0,
                'count': 0
            }
        }
        
        # Scale和bias参数（训练后固定）
        self.scales = {'valence': 1.0, 'arousal': 1.0}
        self.biases = {'valence': 0.0, 'arousal': 0.0}
        
        # 训练状态
        self.is_trained = False
    
    def _map_lexicon_to_range(self, lexicon: Dict) -> Dict:
        """
        将词典值从0~1映射到-1~1
        
        Args:
            lexicon: 原始词典
            
        Returns:
            Dict: 映射后的词典
        """
        mapped = {}
        words = lexicon.get('words', {})
        
        for word, values in words.items():
            mapped[word] = {
                'valence': 2.0 * values['valence'] - 1.0,
                'arousal': 2.0 * values['arousal'] - 1.0
            }
        
        return mapped
    
    def _find_lexicon_words_in_tokens(self, token_ids: List[int], tokenizer) -> Dict[str, List[int]]:
        """
        在token ID序列中找到词典词的位置
        
        Args:
            token_ids: token ID列表
            tokenizer: 分词器
            
        Returns:
            Dict: {word: [token_indices]} 词典词及其在token_ids中的位置
        """
        word_positions = {}
        
        for word in self.lexicon_words:
            # 将词典词转换为token IDs
            try:
                if hasattr(tokenizer, 'encode'):
                    tokenizer_name = type(tokenizer).__name__.lower()
                    if 'glm' in tokenizer_name or 'chatglm' in tokenizer_name:
                        word_token_result = tokenizer([word], return_tensors=None, add_special_tokens=False)
                        if isinstance(word_token_result, dict):
                            word_token_ids = word_token_result.get('input_ids', [])
                            if isinstance(word_token_ids, list) and len(word_token_ids) > 0:
                                if isinstance(word_token_ids[0], list):
                                    word_token_ids = word_token_ids[0]
                        elif isinstance(word_token_result, list) and len(word_token_result) > 0:
                            if isinstance(word_token_result[0], list):
                                word_token_ids = word_token_result[0]
                            else:
                                word_token_ids = word_token_result
                        else:
                            word_token_ids = []
                    else:
                        word_token_ids = tokenizer.encode(word, add_special_tokens=False)
                else:
                    word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                
                if not word_token_ids or len(word_token_ids) == 0:
                    continue
                
                # 处理ChatGLM等tokenizer的空格前缀
                if len(word_token_ids) > 1:
                    first_token_decoded = tokenizer.decode([word_token_ids[0]], skip_special_tokens=True)
                    if not first_token_decoded.strip():
                        word_token_ids = word_token_ids[1:]
                
                if len(word_token_ids) == 0:
                    continue
                
                # 在token_ids中查找匹配
                if len(word_token_ids) == 1:
                    # 单token词，直接查找
                    for i, token_id in enumerate(token_ids):
                        if token_id == word_token_ids[0]:
                            if word not in word_positions:
                                word_positions[word] = []
                            word_positions[word].append(i)
                else:
                    # 多token词，查找连续匹配
                    for i in range(len(token_ids) - len(word_token_ids) + 1):
                        if token_ids[i:i+len(word_token_ids)] == word_token_ids:
                            if word not in word_positions:
                                word_positions[word] = []
                            word_positions[word].append(i)
                            
            except Exception as e:
                continue
        
        return word_positions
    
    def _find_max_projection_in_sentence(self, projections: Dict[str, List[float]], 
                                         token_ids: List[int], tokenizer) -> Dict[str, Tuple[float, float]]:
        """
        在句子中找到每个维度的极大投影值和对应的目标值
        
        Args:
            projections: 投影值字典 {'valence': [...], 'arousal': [...]}
            token_ids: token ID列表
            tokenizer: 分词器
            
        Returns:
            Dict: {dimension: (max_raw_value, target_value)} 每个维度的极大值和目标值
        """
        # 找到句子中的词典词位置
        word_positions = self._find_lexicon_words_in_tokens(token_ids, tokenizer)
        
        if not word_positions:
            return {}
        
        result = {}
        
        for dim in ['valence', 'arousal']:
            max_raw = None
            target_value = None
            
            for word, positions in word_positions.items():
                if word not in self.mapped_lexicon:
                    continue
                
                target = self.mapped_lexicon[word][dim]
                
                # 找到该词所有位置的投影值，取绝对值最大的
                for pos in positions:
                    if pos < len(projections[dim]):
                        raw_value = projections[dim][pos]
                        
                        # 使用绝对值比较，找到最极端的值
                        if max_raw is None or abs(raw_value) > abs(max_raw):
                            max_raw = raw_value
                            target_value = target
            
            if max_raw is not None:
                result[dim] = (max_raw, target_value)
        
        return result
    
    def update_statistics(self, max_values: Dict[str, Tuple[float, float]]):
        """
        更新累积统计量
        
        Args:
            max_values: {dimension: (max_raw_value, target_value)}
        """
        for dim, (max_raw, target) in max_values.items():
            if dim not in self.stats:
                continue
            
            stats = self.stats[dim]
            stats['sum_raw'] += max_raw
            stats['sum_target'] += target
            stats['sum_raw_squared'] += max_raw ** 2
            stats['sum_raw_target'] += max_raw * target
            stats['count'] += 1
    
    def compute_scale_bias(self, dimension: str) -> Tuple[float, float]:
        """
        计算当前维度的scale和bias
        
        Args:
            dimension: 维度名称 ('valence' 或 'arousal')
            
        Returns:
            Tuple: (scale, bias)
        """
        if dimension not in self.stats:
            return 1.0, 0.0
        
        stats = self.stats[dimension]
        
        if stats['count'] < 2:
            # 样本太少，返回默认值
            return 1.0, 0.0
        
        # 使用最小二乘法计算scale和bias
        count = stats['count']
        sum_raw = stats['sum_raw']
        sum_target = stats['sum_target']
        sum_raw_squared = stats['sum_raw_squared']
        sum_raw_target = stats['sum_raw_target']
        
        # 计算scale: (sum_raw_target - sum_raw*sum_target/count) / (sum_raw_squared - sum_raw²/count)
        denominator = sum_raw_squared - (sum_raw ** 2 / count)
        
        if abs(denominator) < 1e-8:
            # 分母太小，返回默认值
            return 1.0, 0.0
        
        numerator = sum_raw_target - (sum_raw * sum_target / count)
        scale = numerator / denominator
        
        # 计算bias: (sum_target - scale * sum_raw) / count
        bias = (sum_target - scale * sum_raw) / count
        
        return scale, bias
    
    def train(self, corpus: List[str], model_loader, projector, verbose=True):
        """
        训练scaler
        
        Args:
            corpus: 训练语料库（每句话至少包含一个词典词）
            model_loader: ModelLoader实例
            projector: EmotionProjector实例（未集成scaler的原始版本）
            verbose: 是否打印训练过程
        """
        print(f"开始训练AdaptiveScaler，语料库大小: {len(corpus)}")
        
        out_of_range_counts = {'valence': 0, 'arousal': 0}
        processed_count = 0
        
        for i, sentence in enumerate(corpus):
            try:
                # 投影获取原始值（训练时不应用缩放）
                projections, tokens = projector.project_text(model_loader, sentence, apply_scaling=False)
                
                # 获取token IDs用于匹配
                token_ids = model_loader.tokenize(sentence, return_tokens=False)
                
                # 找到极大值和目标值
                max_values = self._find_max_projection_in_sentence(projections, token_ids, model_loader.tokenizer)
                
                if not max_values:
                    # 没有找到词典词，跳过
                    continue
                
                # 更新统计量
                self.update_statistics(max_values)
                
                # 立即重新计算scale和bias
                for dim in max_values.keys():
                    scale, bias = self.compute_scale_bias(dim)
                    self.scales[dim] = scale
                    self.biases[dim] = bias
                    
                    # 应用并检查是否超出范围
                    max_raw, _ = max_values[dim]
                    scaled_value = scale * max_raw + bias
                    
                    if abs(scaled_value) > 1.0:
                        out_of_range_counts[dim] += 1
                
                processed_count += 1
                
                # 每100句打印一次进度
                if verbose and (i + 1) % 100 == 0:
                    print(f"  已处理 {i+1}/{len(corpus)} 句，有效样本: {processed_count}")
                    for dim in ['valence', 'arousal']:
                        scale = self.scales[dim]
                        bias = self.biases[dim]
                        count = self.stats[dim]['count']
                        out_count = out_of_range_counts[dim]
                        print(f"    {dim}: scale={scale:.4f}, bias={bias:.4f}, "
                              f"样本数={count}, 超出范围={out_count} ({out_count/max(count,1)*100:.1f}%)")
                
            except Exception as e:
                if verbose:
                    print(f"  警告：处理第{i+1}句时出错: {e}")
                continue
        
        self.is_trained = True
        
        print(f"\n训练完成！")
        print(f"  有效样本数: {processed_count}")
        for dim in ['valence', 'arousal']:
            scale = self.scales[dim]
            bias = self.biases[dim]
            count = self.stats[dim]['count']
            out_count = out_of_range_counts[dim]
            print(f"  {dim}: scale={scale:.4f}, bias={bias:.4f}, "
                  f"样本数={count}, 超出范围={out_count} ({out_count/max(count,1)*100:.1f}%)")
    
    def apply(self, raw_value: float, dimension: str) -> float:
        """
        应用scale和bias，并裁剪到[-1, 1]范围
        
        Args:
            raw_value: 原始投影值
            dimension: 维度名称
            
        Returns:
            float: 缩放后的值（已裁剪到[-1, 1]）
        """
        if dimension not in self.scales:
            return raw_value
        
        scale = self.scales[dimension]
        bias = self.biases[dimension]
        
        # 应用缩放
        scaled_value = scale * raw_value + bias
        
        # 裁剪到[-1, 1]
        clipped_value = max(-1.0, min(1.0, scaled_value))
        
        return clipped_value
    
    def save(self, path: str):
        """
        保存scale和bias到文件
        
        Args:
            path: 保存路径
        """
        data = {
            'scales': self.scales,
            'biases': self.biases,
            'stats': self.stats,
            'is_trained': self.is_trained
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Scaler已保存到: {path}")
    
    def load(self, path: str):
        """
        从文件加载scale和bias
        
        Args:
            path: 文件路径
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Scaler文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.scales = data.get('scales', self.scales)
        self.biases = data.get('biases', self.biases)
        self.stats = data.get('stats', self.stats)
        self.is_trained = data.get('is_trained', False)
        
        print(f"Scaler已从 {path} 加载")
