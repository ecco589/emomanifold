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
    
    def _parse_response(self, raw_response):
        """
        解析模型响应，分离推理过程和实际输出
        
        Args:
            raw_response: 原始响应文本
            
        Returns:
            tuple: (reasoning, output) 推理过程和实际输出
        """
        reasoning = ""
        output = ""
        
        # 先分离推理和输出（在清理标记之前，避免丢失关键标记）
        # DeepSeek-R1 格式: ...推理内容...</think>实际输出
        reasoning_end_marker = "</think>"
        
        if reasoning_end_marker in raw_response:
            # 有明确的结束标记，分离推理和输出
            parts = raw_response.split(reasoning_end_marker, 1)
            reasoning = parts[0].strip()
            output = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 没有明确的推理结束标记，尝试通过内容特征识别
            # 推理过程通常包含：思考性语言、分析过程、计划等
            # 实际输出通常是：直接回答、结论、建议等
            
            # 检查是否包含明显的推理特征（中文）
            reasoning_keywords = [
                "我需要", "我应该", "让我", "首先", "接下来", "然后", "最后",
                "看来", "可能", "应该", "需要", "考虑", "分析", "思考",
                "用户", "问题", "回答", "确保", "检查", "回顾", "今天"
            ]
            
            # 查找可能的推理和输出分界点
            # 通常推理过程会包含多个段落，输出通常是直接回答
            lines = raw_response.split('\n')
            reasoning_lines = []
            output_lines = []
            found_output_start = False
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    if not found_output_start:
                        reasoning_lines.append(line)
                    else:
                        output_lines.append(line)
                    continue
                
                # 检查是否包含推理关键词
                is_reasoning = any(keyword in line_stripped for keyword in reasoning_keywords)
                
                # 如果前几行都是推理，且当前行也是推理，继续作为推理
                if not found_output_start and (is_reasoning or i < 3):
                    reasoning_lines.append(line)
                # 如果遇到看起来像输出的内容（不包含推理关键词，且长度较短或格式不同）
                elif not found_output_start and not is_reasoning and len(line_stripped) < 100:
                    # 可能是输出的开始
                    found_output_start = True
                    output_lines.append(line)
                elif found_output_start:
                    output_lines.append(line)
                else:
                    # 如果推理行数已经很多（>5行），后续可能是输出
                    if len(reasoning_lines) > 5 and not is_reasoning:
                        found_output_start = True
                        output_lines.append(line)
                    else:
                        reasoning_lines.append(line)
            
            reasoning = '\n'.join(reasoning_lines).strip()
            output = '\n'.join(output_lines).strip()
            
            # 如果分离后输出为空，说明整个内容都是推理，将最后一部分作为输出
            if not output and reasoning:
                # 取最后一段作为可能的输出
                reasoning_parts = reasoning.split('\n\n')
                if len(reasoning_parts) > 1:
                    output = reasoning_parts[-1]
                    reasoning = '\n\n'.join(reasoning_parts[:-1])
                else:
                    # 如果只有一段，整个作为输出（可能是没有推理的直接回答）
                    output = reasoning
                    reasoning = ""
        
        # 清理推理过程中的特殊标记
        reasoning = reasoning.replace("<｜Assistant｜>", "")
        reasoning = reasoning.replace("<｜end▁of▁sentence｜>", "")
        reasoning = reasoning.replace("<｜User｜>", "")
        reasoning = reasoning.replace("<｜begin▁of▁sentence｜>", "")
        reasoning = reasoning.replace("<think>", "")
        reasoning = reasoning.replace("</think>", "")
        reasoning = reasoning.replace("<|im_start|>", "").replace("<|im_end|>", "")
        reasoning = reasoning.strip()
        
        # 清理输出中的特殊标记
        output = output.replace("<｜Assistant｜>", "")
        output = output.replace("<｜end▁of▁sentence｜>", "")
        output = output.replace("<｜User｜>", "")
        output = output.replace("<｜begin▁of▁sentence｜>", "")
        output = output.replace("<think>", "")
        output = output.replace("</think>", "")
        output = output.replace("<|im_start|>", "").replace("<|im_end|>", "")
        output = output.strip()
        
        return reasoning, output
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.6, top_p=0.95, do_sample=True, return_reasoning=False, **kwargs):
        """
        生成文本回复
        
        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成的新token数量
            temperature: 采样温度（推荐0.6，根据模型说明）
            top_p: nucleus采样参数（推荐0.95，根据模型说明）
            do_sample: 是否使用采样
            return_reasoning: 是否返回推理过程
            **kwargs: 其他生成参数
            
        Returns:
            如果 return_reasoning=False: 返回实际输出字符串
            如果 return_reasoning=True: 返回 (reasoning, output) 元组
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # 创建 attention_mask
        attention_mask = torch.ones_like(input_ids)
        
        # 获取 eos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.pad_token_id
        
        # 设置 pad_token_id（如果 pad_token_id 和 eos_token_id 相同，使用 eos_token_id）
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        
        # 生成参数（根据模型说明设置）
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "attention_mask": attention_mask,
            "repetition_penalty": 1.1,  # 减少重复
            **kwargs
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **generation_config
            )
        
        # 解码生成的文本（只取新生成的部分）
        generated_ids = outputs[0][input_ids.shape[1]:]
        
        # 移除 eos_token 及其之后的内容
        if eos_token_id in generated_ids:
            eos_index = (generated_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_index) > 0:
                generated_ids = generated_ids[:eos_index[0]]
        
        raw_response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 解析响应，分离推理和输出
        reasoning, output = self._parse_response(raw_response)
        
        if return_reasoning:
            return reasoning, output
        else:
            return output
    
    def chat(self, message, history=None, max_new_tokens=100, temperature=0.6, top_p=0.95, return_reasoning=False, **kwargs):
        """
        与模型进行对话
        
        Args:
            message: 用户消息
            history: 对话历史，格式为 [(user_msg1, assistant_msg1), (user_msg2, assistant_msg2), ...]
            max_new_tokens: 最大生成的新token数量
            temperature: 采样温度（默认0.6，根据模型说明推荐）
            top_p: nucleus采样参数（默认0.95，根据模型说明推荐）
            return_reasoning: 是否返回推理过程
            **kwargs: 其他生成参数
            
        Returns:
            如果 return_reasoning=False: 返回实际输出字符串
            如果 return_reasoning=True: 返回 (reasoning, output) 元组
        """
        if not self._model_loaded:
            raise RuntimeError("请先调用 load() 方法加载模型")
        
        if history is None:
            history = []
        
        # 构建消息列表用于 chat_template
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": message})
        
        # 使用 tokenizer 的 chat_template 格式化对话
        # 根据模型说明：避免添加系统提示，所有指令都应包含在用户提示中
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # 使用 chat_template 格式化，并添加 generation prompt
            # 根据模型说明，强制模型在输出开始时使用 "\n" 来确保进行推理
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 回退到手动格式化（使用 DeepSeek-R1 的特殊格式）
            conversation_text = ""
            for msg in messages:
                if msg["role"] == "user":
                    conversation_text += f"<｜User｜>{msg['content']}<｜end▁of▁sentence｜>"
                elif msg["role"] == "assistant":
                    conversation_text += f"<｜Assistant｜>{msg['content']}<｜end▁of▁sentence｜>"
            # 添加 assistant 提示，强制使用 "\n" 开始推理
            prompt = conversation_text + "<｜Assistant｜>\n"
        
        # 生成回复
        result = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_reasoning=return_reasoning,
            **kwargs
        )
        
        return result