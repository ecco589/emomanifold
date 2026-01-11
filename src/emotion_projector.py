"""
Token-level 情感投影模块
将每个 Token 的隐藏状态投影到情感维度上
"""
import torch
from typing import List, Dict, Tuple, Optional
import config


class EmotionProjector:
    """情感投影器"""
    
    def __init__(self, dimension_vectors: Dict[str, torch.Tensor], device=None):
        """
        初始化投影器
        
        Args:
            dimension_vectors: 情感维度向量字典，如 {'valence': tensor, 'arousal': tensor}
            device: 设备类型
        """
        self.dimensions = dimension_vectors
        self.device = device or config.DEVICE
        
        # 将维度向量移到指定设备并确保是float类型
        for key in self.dimensions:
            self.dimensions[key] = self.dimensions[key].to(self.device).float()
    
    def project_hidden_state(self, hidden_state: torch.Tensor) -> Dict[str, float]:
        """
        将单个隐藏状态向量投影到情感维度
        
        Args:
            hidden_state: 隐藏状态向量
            
        Returns:
            Dict: 各维度的投影值
        """
        if not isinstance(hidden_state, torch.Tensor):
            hidden_state = torch.tensor(hidden_state)
        
        hidden_state = hidden_state.to(self.device).float()
        
        projections = {}
        for dim_name, dim_vector in self.dimensions.items():
            # 计算点积（投影值）
            projection_value = torch.dot(hidden_state, dim_vector).item()
            projections[dim_name] = projection_value
        
        return projections
    
    def project_sequence(self, hidden_states: torch.Tensor) -> Dict[str, List[float]]:
        """
        将序列的隐藏状态投影到情感维度
        
        Args:
            hidden_states: 隐藏状态序列
            
        Returns:
            Dict: 每个维度的时间序列
        """
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states)
        
        hidden_states = hidden_states.to(self.device).float()
        
        # 处理batch维度
        if hidden_states.dim() == 3:
            # [batch_size, seq_len, hidden_dim] -> 取第一个batch
            hidden_states = hidden_states[0]
        elif hidden_states.dim() != 2:
            raise ValueError(f"隐藏状态维度应为2或3，得到: {hidden_states.dim()}")
        
        # [seq_len, hidden_dim]
        seq_len = hidden_states.shape[0]
        
        projections = {dim_name: [] for dim_name in self.dimensions.keys()}
        
        # 对每个时间步进行投影
        for t in range(seq_len):
            hidden_state_t = hidden_states[t]  # [hidden_dim]
            proj_t = self.project_hidden_state(hidden_state_t)
            
            for dim_name, value in proj_t.items():
                projections[dim_name].append(value)
        
        return projections
    
    def project_token_level(self, hidden_states: torch.Tensor, 
                           token_ids: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """
        从隐藏状态中提取每个 Token 的情感坐标
        
        Args:
            hidden_states: 隐藏状态
            token_ids: Token ID 列表（可选）
            
        Returns:
            Dict: Token-level 情感坐标
        """
        return self.project_sequence(hidden_states)
    
    def project_text(self, model_loader, text: str) -> Tuple[Dict[str, List[float]], List[str]]:
        """
        对文本进行完整的情感投影流程
        
        Args:
            model_loader: ModelLoader 实例
            text: 输入文本
            
        Returns:
            Tuple: (情感坐标字典, token列表)
        """
        # 分词
        token_ids, tokens = model_loader.tokenize(text, return_tokens=True)
        
        # 获取隐藏状态
        hidden_states = model_loader.get_hidden_states(
            torch.tensor([token_ids]), 
            return_dict=False
        )  # [batch_size, seq_len, hidden_dim]
        
        # 投影到情感维度
        emotion_coords = self.project_token_level(hidden_states, token_ids)
        
        return emotion_coords, tokens
