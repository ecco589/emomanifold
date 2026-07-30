#!/usr/bin/env python3
"""
E1：Qwen 梯度 Spearman（极词 N=10..80，中/英）。

英文 Spearman 使用 NIPS 五模型交集 300 词（与 fig2 阴影带可比）；
极词筛选不排除 hold-out（对齐 NIPS B1）。

用法（仓库根目录）:
  python RAL/experiment/A/A1/code/build_holdout.py   # 仅中文 hold-out
  python RAL/experiment/A/A1/code/run_e1.py
  python RAL/experiment/A/A1/code/plot_e1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
E1_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from project_root import find_project_root

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

POLE_NS = [10, 20, 30, 40, 50, 60, 70, 80]
LANGUAGES = ("en", "zh")


def _load_model():
    from src.model_loader import CausalModelLoader

    loader = CausalModelLoader("qwen", device="cpu")
    loader.load()
    return loader


def _ensure_zh_holdout(result_dir: Path):
    from build_holdout import main as build_main

    zh_path = result_dir / "holdout_zh.json"
    if not zh_path.is_file():
        print("中文 hold-out 不存在，正在构建...")
        build_main()


def main():
    from b1_lexicon import get_priority, load_language_nrc
    from e1_core import run_single_pole_n
    from holdout_loader import load_spearman_holdout, validate_qwen_single_token

    result_dir = E1_DIR / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("加载 Qwen (Instruct)...")
    loader = _load_model()
    _ensure_zh_holdout(result_dir)

    en_holdout = load_spearman_holdout("en", result_dir)
    en_check = validate_qwen_single_token(en_holdout, loader.tokenizer)
    print(
        "英文 hold-out: NIPS 五模型交集 300 词; "
        f"Qwen 单 token 覆盖 {en_check['n_per_axis']}"
    )
    if not en_check["all_single_token"]:
        print(f"  警告: 非单 token {len(en_check['skipped'])} 个，Spearman 将跳过这些词")

    import config as _cfg

    _qwen = _cfg.ModelConfig("qwen")
    summary = {
        "languages": {},
        "pole_ns": POLE_NS,
        "model": "qwen",
        "model_id": _qwen.model_id,
        "en_holdout": "data/nips_holdout_intersection_en.json",
        "pole_holdout_disjoint": False,
    }

    for lang in LANGUAGES:
        print(f"\n{'='*60}\n语言: {lang}\n{'='*60}")
        nrc = load_language_nrc(lang, PROJECT_ROOT)
        priority = get_priority(lang, nrc)
        holdout = load_spearman_holdout(lang, result_dir)

        lang_results = {}
        for pole_n in POLE_NS:
            t0 = time.time()
            out_dir = result_dir / lang / f"pole_{pole_n}"
            print(f"  pole_n={pole_n} ...", end=" ", flush=True)
            try:
                report = run_single_pole_n(
                    loader,
                    nrc,
                    priority,
                    holdout,
                    pole_n,
                    lang,
                    out_dir,
                    exclude=set(),
                )
                sp = report["spearman"]

                def _fmt_rho(x):
                    return "—" if x is None else f"{x:.4f}"

                print(
                    f"ρ_V={_fmt_rho(sp['valence']['rho'])} "
                    f"ρ_A={_fmt_rho(sp['arousal']['rho'])} "
                    f"ρ_D={_fmt_rho(sp['dominance']['rho'])} "
                    f"({time.time()-t0:.1f}s)"
                )
                lang_results[str(pole_n)] = sp
            except Exception as e:
                print(f"FAILED: {e}")
                lang_results[str(pole_n)] = {"error": str(e)}

        summary["languages"][lang] = lang_results

    summary_path = result_dir / "e1_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")
    print("Run export_pole_sweep_series.py && plot_e1.py to regenerate figures.")


if __name__ == "__main__":
    main()
