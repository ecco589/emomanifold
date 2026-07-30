#!/usr/bin/env python3
"""构建 E1 中英文 hold-out 词表（各 300 词，Qwen 单 token）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from project_root import find_project_root

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from b1_lexicon import (  # noqa: E402
    build_holdout_lexicon,
    get_priority,
    load_language_nrc,
)
from transformers import AutoTokenizer  # noqa: E402
import config  # noqa: E402


def main():
    result_dir = CODE_DIR.parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config.ModelConfig("qwen")
    model_path = str(model_cfg.local_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    for lang in ("en", "zh"):
        print(f"\n=== Hold-out: {lang} ===")
        nrc = load_language_nrc(lang, PROJECT_ROOT)
        priority = get_priority(lang, nrc)
        holdout = build_holdout_lexicon(tokenizer, nrc, lang)
        out = result_dir / f"holdout_{lang}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(holdout, f, indent=2, ensure_ascii=False)
        print(f"  words: {len(holdout['words'])}")
        print(f"  per axis: {[len(holdout['per_axis'][d]) for d in holdout['axis_names']]}")
        print(f"  wrote: {out}")

        # 保存扩充后的 priority 供检查
        pri_out = result_dir / f"priority_{lang}.json"
        with pri_out.open("w", encoding="utf-8") as f:
            json.dump(priority, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
