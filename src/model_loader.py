"""
模型加载模块
负责加载 DeepSeek-R1-Distill-Qwen-1.5B 模型并提取必要的组件
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_name=None, device=None):
        """
        初始化模型加载器
        
        Args:
            model_name: 模型名称或路径，默认使用 config.MODEL_NAME
            device: 设备类型，默认使用 config.DEVICE
        """
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or config.DEVICE
        self.model = None
        self.tokenizer = None
        self.output_embedding = None
        self._model_loaded = False
        
    def load(self):
        """加载模型和分词器"""
        if self._model_loaded:
            print("模型已加载，跳过重复加载")
            return
        
        print(f"正在加载模型: {self.model_name}")
        print(f"设备: {self.device}")
        
        try:
            # 优先使用本地模型路径
            model_path = self.model_name
            if hasattr(config, 'MODEL_LOCAL_PATH') and config.MODEL_LOCAL_PATH.exists():
                model_path = str(config.MODEL_LOCAL_PATH)
                print(f"使用本地模型路径: {model_path}")
            
            # 加载分词器
            print("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True  # 只使用本地文件，避免网络请求
            )
            
            # 加载模型
            print("加载模型（这可能需要一些时间）...")
            # 对于MPS设备，不使用device_map，直接加载后移动到设备
            use_device_map = False
            if self.device == "cuda":
                try:
                    import accelerate
                    use_device_map = True
                except ImportError:
                    use_device_map = False
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if use_device_map else None,
                output_hidden_states=True,  # 需要输出隐藏状态
                local_files_only=True  # 只使用本地文件，避免网络请求
            )
            
            # 如果没有使用device_map，手动移动到设备
            if not use_device_map:
                self.model = self.model.to(self.device)
            
            self.model.eval()  # 设置为评估模式
            
            # 提取输出层权重矩阵（解嵌入矩阵）
            if hasattr(self.model, 'lm_head'):
                self.output_embedding = self.model.lm_head.weight.data
            elif hasattr(self.model, 'embed_out'):
                self.output_embedding = self.model.embed_out.weight.data
            else:
                # 尝试从config中获取输出层名称
                print("警告：未找到标准的输出层，尝试查找...")
                for name, module in self.model.named_modules():
                    if 'lm_head' in name.lower() or 'output' in name.lower():
                        if hasattr(module, 'weight'):
                            self.output_embedding = module.weight.data
                            print(f"找到输出层: {name}")
                            break
            
            if self.output_embedding is None:
                raise ValueError("无法找到输出层权重矩阵")
            
            print(f"模型加载成功")
            print(f"   词汇表大小: {len(self.tokenizer)}")
            print(f"   隐藏维度: {self.output_embedding.shape[1]}")
            print(f"   输出层权重形状: {self.output_embedding.shape}")
            
            self._model_loaded = True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def get_output_embedding_matrix(self):
        """
        获取输出层权重矩阵（解嵌入矩阵）
        
        Returns:
            torch.Tensor: 形状为 [vocab_size, hidden_dim] 的权重矩阵
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        if self.output_embedding is None:
            raise RuntimeError("输出层权重矩阵未找到")
        
        return self.output_embedding
    
    def get_hidden_states(self, input_ids, return_dict=True):
        """
        获取指定输入的隐藏状态
        
        Args:
            input_ids: 输入 token IDs，形状为 [batch_size, seq_len] 或 [seq_len]
            return_dict: 是否返回字典格式
            
        Returns:
            如果 return_dict=True: 返回包含 hidden_states 的字典
            如果 return_dict=False: 返回最后一层的隐藏状态 [batch_size, seq_len, hidden_dim]
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        # 确保 input_ids 是 tensor 且在正确的设备上
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        # 添加 batch 维度（如果没有）
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        
        # 前向传播，获取隐藏状态
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        if return_dict:
            return {
                'hidden_states': hidden_states,
                'last_hidden_state': hidden_states,
                'all_hidden_states': outputs.hidden_states
            }
        else:
            return hidden_states
    
    def encode(self, text):
        """
        将文本编码为 token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: token IDs
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        return self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
    
    def decode(self, token_ids):
        """
        将 token IDs 解码为文本
        
        Args:
            token_ids: token IDs（可以是 tensor 或 list）
            
        Returns:
            str: 解码后的文本
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def tokenize(self, text, return_tokens=False):
        """
        分词并返回 token IDs 和可选的 token 字符串列表
        
        Args:
            text: 输入文本
            return_tokens: 是否返回 token 字符串列表
            
        Returns:
            如果 return_tokens=False: 返回 token IDs (list)
            如果 return_tokens=True: 返回 (token_ids, tokens) 元组，tokens是优化后的可读字符串列表
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        # 获取原始tokens
        raw_tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(raw_tokens)
        
        if return_tokens:
            # 优化token显示：将tokens转换为可读的中文文本
            readable_tokens = self._convert_tokens_to_readable(raw_tokens, token_ids)
            return token_ids, readable_tokens
        else:
            return token_ids
    
    def _convert_tokens_to_readable(self, raw_tokens, token_ids):
        """
        将原始tokens转换为可读的文本
        
        Args:
            raw_tokens: 原始token列表
            token_ids: token ID列表
            
        Returns:
            List[str]: 可读的token字符串列表
        """
        readable_tokens = []
        
        for raw_token, token_id in zip(raw_tokens, token_ids):
            try:
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if decoded and len(decoded.strip()) > 0:
                    decoded_clean = decoded.strip()
                    if len(decoded_clean) <= 10:
                        readable_tokens.append(decoded_clean)
                        continue
            except Exception:
                pass
            
            token_clean = raw_token
            if token_clean.startswith('##'):
                token_clean = token_clean[2:]
            elif token_clean.startswith('Ġ'):
                token_clean = token_clean[1:]
            elif token_clean.startswith('▁'):
                token_clean = token_clean[1:]
            
            if token_clean in ['<unk>', '<pad>', '<s>', '</s>', '<|endoftext|>', '<|im_start|>', '<|im_end|>']:
                readable_tokens.append(token_clean)
            elif any('\u4e00' <= char <= '\u9fff' for char in token_clean):
                readable_tokens.append(token_clean)
            elif token_clean.isprintable() and len(token_clean) <= 20:
                readable_tokens.append(token_clean)
            else:
                try:
                    decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    if decoded and len(decoded) <= 10:
                        readable_tokens.append(decoded)
                    else:
                        readable_tokens.append(f"[{token_id}]")
                except:
                    readable_tokens.append(f"[{token_id}]")
        
        return readable_tokens
