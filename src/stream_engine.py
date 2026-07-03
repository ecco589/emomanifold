"""
流式 Chat LM（ERNIE / Qwen）+ 逐 token VAD 投影 + 表情权重。
供 stream_ernie_vad_ui 等复用。默认模型见 config.DEFAULT_STREAM_MODEL，可用 EMOTION_MODEL 切换。
"""
from typing import Iterator, Tuple, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.adaptive_scaler import AdaptiveScaler
from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.fkcn import FKCN
from src.grimace_utils import StreamingEMASmoother


def generate_with_vad_stream(
    model,
    tokenizer,
    projector: EmotionProjector,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    device: str = "cpu",
) -> Iterator[Tuple[str, Dict[str, float]]]:
    """
    流式生成，每产生一个内容 token 就 yield (token_str, vad_dict)。
    使用因果 LM：hidden_states[:, -1] 是预测下一个 token 的上下文表示，用于该 token 的 VAD。
    """
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

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id = logits.argmax(dim=-1).item()

        if next_token_id == eos_id:
            break

        hidden_np = last_hidden[0].float().cpu()
        vad = projector.project_hidden_state(hidden_np, apply_scaling=True)
        token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)

        yield token_str, vad

        next_tensor = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
        ], dim=1)


def generate_with_face_stream(
    model,
    tokenizer,
    projector: EmotionProjector,
    messages: List[Dict[str, str]],
    fkcn: FKCN,
    *,
    ema_alpha: float = 0.6,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    device: str = "cpu",
) -> Iterator[Tuple[str, Dict[str, float], List[float]]]:
    """流式生成 + 逐 token FKCN 7 维权重（EMA 平滑）。Yields: (token_str, vad_dict, fkcn_7d)"""
    ema = StreamingEMASmoother(alpha=ema_alpha, initial=[0.0] * 7)
    for token_str, vad in generate_with_vad_stream(
        model, tokenizer, projector, messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    ):
        v, a = vad.get("valence", 0), vad.get("arousal", 0)
        fkcn_raw = fkcn.compute_weights_batch([(v, a)])[0]
        smoothed = ema.update(list(fkcn_raw))
        yield token_str, vad, smoothed


def load_chat_model_and_projector(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
):
    """加载 CausalLM 以及 VAD 投影器（ERNIE / Qwen chat 场景）。"""
    if model_name is None:
        model_name = config.get_active_model()
    model_config = config.ModelConfig(model_name)
    if not model_config.is_available():
        raise FileNotFoundError(f"模型不可用: {model_config.local_path}")

    model_path = str(model_config.local_path)
    if device is None:
        device = "cpu"
    if device == "mps" and model_name in ("ernie",):
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

    if hasattr(model, "get_input_embeddings"):
        output_embedding = model.get_input_embeddings().weight.data
    elif hasattr(model, "lm_head"):
        output_embedding = model.lm_head.weight.data
    else:
        raise ValueError("无法获取 embedding 矩阵")

    dimension_extractor = DimensionExtractor(output_embedding, tokenizer, device=device)
    vad_lexicon = dimension_extractor.load_vad_lexicon(model_name=model_name)
    dimensions = dimension_extractor.extract_all_dimensions(
        vad_lexicon, dimensions=model_config.emotion_dimensions
    )

    scaler_path = config.PROJECT_ROOT / "outputs" / "scalers" / f"{model_name}_scaler.json"
    scaler = AdaptiveScaler(vad_lexicon)
    scaler.load(str(scaler_path))

    projector = EmotionProjector(dimensions, scaler=scaler, device=device)

    return model, tokenizer, projector


def load_fkcn(device: str = "cpu") -> FKCN:
    return FKCN(device=device)
