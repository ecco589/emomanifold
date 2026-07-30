"""E1 Spearman hold-out 加载（英文用 NIPS 五模型交集，与阴影带可比）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

CODE_DIR = Path(__file__).resolve().parent
E1_DIR = CODE_DIR.parent
DATA_DIR = E1_DIR / "data"
RESULT_DIR = E1_DIR / "result"

NIPS_HOLDOUT_EN = DATA_DIR / "nips_holdout_intersection_en.json"


def load_spearman_holdout(language: str, result_dir: Optional[Path] = None) -> dict:
    """
    - en: NIPS reform 五模型 tokenizer 交集 300 词（固定文献词表）
    - zh: 本地 Qwen hold-out（不参与 NIPS 英文阴影带对比）
    """
    result_dir = result_dir or RESULT_DIR
    if language == "en":
        path = NIPS_HOLDOUT_EN
    elif language == "zh":
        path = result_dir / "holdout_zh.json"
    else:
        raise ValueError(language)

    if not path.is_file():
        raise FileNotFoundError(f"Spearman hold-out missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def validate_qwen_single_token(holdout: dict, tokenizer) -> Dict[str, Any]:
    """统计 hold-out 在 Qwen 分词器下的单 token 覆盖率。"""
    per_axis = holdout["per_axis"]
    dims = holdout.get("axis_names", ("valence", "arousal", "dominance"))
    skipped = []
    ok_per_dim = {}
    for dim in dims:
        ok = 0
        for word in per_axis[dim]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                skipped.append(f"{dim}:{word}")
            else:
                ok += 1
        ok_per_dim[dim] = ok
    return {
        "n_per_axis": ok_per_dim,
        "skipped": skipped,
        "all_single_token": len(skipped) == 0,
    }
