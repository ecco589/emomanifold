#!/usr/bin/env python3
"""
M1: 流式 ERNIE 聊天 + 逐 token VAD 投影
文字输入 → ERNIE 流式输出 → 终端逐 token 显示 VAD

核心逻辑：generate_with_vad_stream() 可被 UI 等模块复用。
"""
import sys
from pathlib import Path
from typing import Iterator, Tuple, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.adaptive_scaler import AdaptiveScaler


# ---------------------------------------------------------------------------
# 步骤 1: 流式生成核心逻辑
# ---------------------------------------------------------------------------

def generate_with_vad_stream(
    model,
    tokenizer,
    projector: EmotionProjector,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cpu",
) -> Iterator[Tuple[str, Dict[str, float]]]:
    """
    流式生成，每产生一个内容 token 就 yield (token_str, vad_dict)。
    使用因果 LM：hidden_states[:, -1] 是预测下一个 token 的上下文表示，用于该 token 的 VAD。

    Yields:
        (token_str, vad_dict): token_str 为解码后的字符串，vad_dict 为 {"valence": v, "arousal": a}
    """
    # 构建 chat prompt
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [text], add_special_tokens=False, return_tensors="pt"
    ).to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id or 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        logits = outputs.logits[:, -1, :]
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # 采样
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id = logits.argmax(dim=-1).item()

        if next_token_id == eos_id:
            break

        # 投影到 VAD（使用预测该 token 的 hidden state）
        hidden_np = last_hidden[0].float().cpu()
        vad = projector.project_hidden_state(hidden_np, apply_scaling=True)
        token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)

        yield token_str, vad

        # 拼接到序列
        next_tensor = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
        ], dim=1)


# ---------------------------------------------------------------------------
# 步骤 2: 模型加载（CausalLM 用于 chat 生成）
# ---------------------------------------------------------------------------

def load_chat_model_and_projector(
    model_name: str = "ernie",
    device: Optional[str] = None,
):
    """
    加载用于 chat 的 CausalLM 以及 VAD 投影器。
    ERNIE 在 chat 场景使用 AutoModelForCausalLM，与投影用的 AutoModel 不同。
    """
    model_config = config.ModelConfig(model_name)
    if not model_config.is_available():
        raise FileNotFoundError(f"模型不可用: {model_config.local_path}")

    model_path = str(model_config.local_path)
    if device is None:
        device = "cpu"  # ERNIE 在 MPS 上有兼容问题
    if device == "mps" and model_name == "ernie":
        device = "cpu"

    print(f"加载模型: {model_path}")
    print(f"设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()

    # 获取 embedding 矩阵用于维度提取（与 ModelLoader 中双向模型逻辑一致）
    if hasattr(model, "get_input_embeddings"):
        output_embedding = model.get_input_embeddings().weight.data
    elif hasattr(model, "lm_head"):
        output_embedding = model.lm_head.weight.data
    else:
        raise ValueError("无法获取 embedding 矩阵")

    # 维度提取 + Scaler + 投影器
    dimension_extractor = DimensionExtractor(output_embedding, tokenizer, device=device)
    vad_lexicon = dimension_extractor.load_vad_lexicon()
    dimensions = dimension_extractor.extract_all_dimensions(vad_lexicon)

    scaler_path = project_root / "outputs" / "scalers" / f"{model_name}_scaler.json"
    scaler = AdaptiveScaler(vad_lexicon)
    scaler.load(str(scaler_path))

    projector = EmotionProjector(dimensions, scaler=scaler, device=device)

    return model, tokenizer, projector


# ---------------------------------------------------------------------------
# 步骤 3: 终端交互
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("M1: 流式 ERNIE 聊天 + 逐 token VAD 投影")
    print("=" * 60)

    model, tokenizer, projector = load_chat_model_and_projector("ernie")
    device = next(model.parameters()).device

    print("\n模型加载完成。输入消息，空行或 q 退出\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input or user_input.lower() == "q":
            break

        messages = [{"role": "user", "content": user_input}]

        print("ERNIE: ", end="", flush=True)
        vad_parts = []

        for token_str, vad in generate_with_vad_stream(
            model, tokenizer, projector, messages, device=str(device)
        ):
            print(token_str, end="", flush=True)
            v, a = vad.get("valence", 0), vad.get("arousal", 0)
            vad_parts.append(f"{token_str}(V:{v:.2f},A:{a:.2f})")

        print()
        print("  VAD: ", " ".join(vad_parts))
        print()

    print("退出。")


if __name__ == "__main__":
    main()
