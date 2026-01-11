"""
情感维度提取模块
从解嵌入矩阵中提取 Valence 和 Arousal 等情感维度
"""
import torch
import json
from typing import List, Dict, Optional
from pathlib import Path
import config


class DimensionExtractor:
    """情感维度提取器"""
    
    def __init__(self, output_embedding_matrix, tokenizer, device=None):
        """
        初始化维度提取器
        
        Args:
            output_embedding_matrix: 输出层权重矩阵 [vocab_size, hidden_dim]
            tokenizer: 分词器，用于将词转换为 token IDs
            device: 设备类型
        """
        self.output_embedding = output_embedding_matrix
        self.tokenizer = tokenizer
        self.device = device or config.DEVICE
        self.dimensions = {}  # 存储提取的情感维度向量
        
        # 将embedding移到指定设备
        if isinstance(self.output_embedding, torch.Tensor):
            self.output_embedding = self.output_embedding.to(self.device)
        
    def load_vad_lexicon(self, lexicon_path=None):
        """
        加载 VAD 词库
        
        Args:
            lexicon_path: 词库文件路径，默认使用config中的路径
            
        Returns:
            Dict: 包含正负效价词、高低唤醒度词的字典
        """
        if lexicon_path is None:
            lexicon_path = config.VAD_LEXICON_DIR / "vad_lexicon_chinese.json"
        
        lexicon_path = Path(lexicon_path)
        
        if not lexicon_path.exists():
            raise FileNotFoundError(f"词库文件不存在: {lexicon_path}")
        
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            lexicon = json.load(f)
        
        return lexicon
    
    def get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """
        获取单个词在输出层权重矩阵中的embedding
        
        Args:
            word: 输入词
            
        Returns:
            torch.Tensor: 词的embedding向量，如果词不存在则返回None
        """
        try:
            # 将词转换为token IDs
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            
            if len(token_ids) == 0:
                return None
            
            # 取第一个token的ID（如果词被分成多个token）
            token_id = token_ids[0]
            
            # 获取对应的embedding
            if token_id < self.output_embedding.shape[0]:
                embedding = self.output_embedding[token_id]
                return embedding.to(self.device)
            else:
                return None
                
        except Exception as e:
            print(f"警告：无法获取词 '{word}' 的embedding: {e}")
            return None
    
    def get_word_cluster_embedding(self, words: List[str]) -> torch.Tensor:
        """
        获取词簇的平均embedding
        
        Args:
            words: 词列表
            
        Returns:
            torch.Tensor: 平均embedding向量 [hidden_dim]
        """
        embeddings = []
        valid_words = []
        
        for word in words:
            emb = self.get_word_embedding(word)
            if emb is not None:
                embeddings.append(emb)
                valid_words.append(word)
        
        if len(embeddings) == 0:
            raise ValueError(f"词簇中没有有效的词: {words}")
        
        if len(valid_words) < len(words):
            print(f"警告：{len(words) - len(valid_words)} 个词未找到对应的embedding")
        
        # 计算平均embedding
        embeddings_tensor = torch.stack(embeddings)  # [n_words, hidden_dim]
        mean_embedding = embeddings_tensor.mean(dim=0)  # [hidden_dim]
        
        return mean_embedding
    
    def extract_valence_dimension(self, positive_words: List[str], 
                                   negative_words: List[str]) -> torch.Tensor:
        """
        提取 Valence（效价）维度
        
        Args:
            positive_words: 高效价词列表
            negative_words: 低效价词列表
            
        Returns:
            torch.Tensor: Valence 维度向量
        """
        print(f"提取 Valence 维度...")
        print(f"  高效价词: {len(positive_words)} 个")
        print(f"  低效价词: {len(negative_words)} 个")
        
        # 获取正负词簇的平均embedding
        pos_embedding = self.get_word_cluster_embedding(positive_words)
        neg_embedding = self.get_word_cluster_embedding(negative_words)
        
        # 计算差值向量（从负到正的方向）
        valence_dim = pos_embedding - neg_embedding
        
        # 归一化
        valence_dim = valence_dim / (torch.norm(valence_dim) + 1e-8)
        
        print(f"  Valence 维度提取完成，向量范数: {torch.norm(valence_dim):.4f}")
        
        return valence_dim
    
    def extract_arousal_dimension(self, high_words: List[str], 
                                  low_words: List[str]) -> torch.Tensor:
        """
        提取 Arousal（唤醒度）维度
        
        Args:
            high_words: 高唤醒度词列表
            low_words: 低唤醒度词列表
            
        Returns:
            torch.Tensor: Arousal 维度向量 [hidden_dim]
        """
        print(f"提取 Arousal 维度...")
        print(f"  高唤醒度词: {len(high_words)} 个")
        print(f"  低唤醒度词: {len(low_words)} 个")
        
        # 获取高低唤醒度词簇的平均embedding
        high_embedding = self.get_word_cluster_embedding(high_words)
        low_embedding = self.get_word_cluster_embedding(low_words)
        
        # 计算差值向量（从低到高的方向）
        arousal_dim = high_embedding - low_embedding
        
        # 归一化
        arousal_dim = arousal_dim / (torch.norm(arousal_dim) + 1e-8)
        
        print(f"  Arousal 维度提取完成，向量范数: {torch.norm(arousal_dim):.4f}")
        
        return arousal_dim
    
    def extract_all_dimensions(self, vad_lexicon: Dict) -> Dict[str, torch.Tensor]:
        """
        提取所有情感维度
        
        Args:
            vad_lexicon: VAD 词库字典
            
        Returns:
            Dict: 包含所有维度向量的字典
        """
        word_groups = vad_lexicon.get('word_groups', {})
        
        # 提取 Valence 维度
        valence_dim = self.extract_valence_dimension(
            word_groups.get('valence_positive', []),
            word_groups.get('valence_negative', [])
        )
        self.dimensions['valence'] = valence_dim
        
        # 提取 Arousal 维度
        arousal_dim = self.extract_arousal_dimension(
            word_groups.get('arousal_high', []),
            word_groups.get('arousal_low', [])
        )
        self.dimensions['arousal'] = arousal_dim
        
        return self.dimensions
