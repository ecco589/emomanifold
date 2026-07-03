"""
从主词库中按模型分词器筛选单 token 极词，生成 model-specific word_groups。

流程：
1. 读取主词库 vad_lexicon_chinese.json（words 含 V/A/D）
2. 保留 encode 后为恰好 1 个 token 的词
3. 六极各取 top-N（默认 10），同极内不重复；跨极允许复用（与 ERNIE 现有词库一致）
4. 写出 vad_lexicon_chinese_{model}.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from transformers import AutoTokenizer


POLE_SPECS: List[Tuple[str, str, bool]] = [
    ("valence_positive", "valence", True),
    ("valence_negative", "valence", False),
    ("arousal_high", "arousal", True),
    ("arousal_low", "arousal", False),
    ("dominance_high", "dominance", True),
    ("dominance_low", "dominance", False),
]


def is_single_token(word: str, tokenizer) -> bool:
    ids = tokenizer.encode(word, add_special_tokens=False)
    return len(ids) == 1


def filter_single_token_words(words: Dict, tokenizer) -> Dict[str, Dict]:
    result = {}
    for word, values in words.items():
        if is_single_token(word, tokenizer):
            result[word] = values
    return result


def select_pole_words(
    candidates: Dict[str, Dict],
    dim: str,
    descending: bool,
    count: int,
) -> List[str]:
    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1].get(dim, 0.5),
        reverse=descending,
    )
    selected: List[str] = []
    seen = set()
    for word, _ in ranked:
        if word in seen:
            continue
        selected.append(word)
        seen.add(word)
        if len(selected) >= count:
            break
    return selected


def build_word_groups(
    single_token_words: Dict[str, Dict],
    poles_per_dim: int = 10,
    include_dominance: bool = True,
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    specs = POLE_SPECS if include_dominance else POLE_SPECS[:4]

    for pole_name, dim, descending in specs:
        groups[pole_name] = select_pole_words(
            single_token_words, dim, descending, poles_per_dim
        )
    return groups


def resolve_model_path(model_name: str) -> Path:
    model_config = config.ModelConfig(model_name)
    if not model_config.is_available():
        raise FileNotFoundError(f"模型不可用: {model_config.local_path}")
    return model_config.local_path


def extract_for_model(
    model_name: str,
    master_path: Path = None,
    output_path: Path = None,
    poles_per_dim: int = 10,
    include_dominance: bool = None,
) -> Path:
    master_path = master_path or (config.VAD_LEXICON_DIR / "vad_lexicon_chinese.json")
    if include_dominance is None:
        include_dominance = model_name == "qwen"

    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    model_path = resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )

    words = master.get("words", {})
    single_token_words = filter_single_token_words(words, tokenizer)
    word_groups = build_word_groups(
        single_token_words, poles_per_dim, include_dominance
    )

    out = {
        "description": f"中文VAD情感词库 - {model_name} 单token极词",
        "model": model_name,
        "format": master.get("format", {}),
        "words": {w: words[w] for w in single_token_words},
        "word_groups": word_groups,
        "meta": {
            "master_lexicon": str(master_path.name),
            "single_token_count": len(single_token_words),
            "poles_per_dim": poles_per_dim,
            "include_dominance": include_dominance,
        },
    }

    if output_path is None:
        output_path = config.VAD_LEXICON_DIR / f"vad_lexicon_chinese_{model_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"模型: {model_name}")
    print(f"单 token 候选词: {len(single_token_words)} / {len(words)}")
    for pole, items in word_groups.items():
        print(f"  {pole}: {len(items)} -> {items}")
    print(f"已写入: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="提取模型单 token 情感极词")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ernie", "qwen", "deepseek", "glm"],
        help="目标模型",
    )
    parser.add_argument("--poles", type=int, default=10, help="每极词数")
    parser.add_argument(
        "--no-dominance",
        action="store_true",
        help="不生成 dominance 极（ERNIE 默认 2D）",
    )
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    args = parser.parse_args()

    include_d = not args.no_dominance
    if args.model == "ernie":
        include_d = False

    extract_for_model(
        args.model,
        poles_per_dim=args.poles,
        include_dominance=include_d,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
