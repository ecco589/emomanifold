#!/usr/bin/env python3
"""
B1 小批量烟雾测试：10 对 iSarcasmEval SubTask C 句子，纯前向，不写文件。

用法（仓库根目录）:
  python RAL/experiment/B/B1/code/run_b1_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
B1_DIR = CODE_DIR.parent
E1_CODE = B1_DIR.parent.parent / "A" / "A1" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(E1_CODE))

from project_root import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from b1_lexicon import load_language_nrc  # noqa: E402
from b1_core import (  # noqa: E402
    annotate_surface_valence,
    load_e1_projector,
    load_isarcasm_task_c,
    run_pair_batch,
    select_positive_surface_pairs,
)

N_PAIRS = 10
DATA_CSV = B1_DIR / "data" / "task_C_En_test.csv"


def main():
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f"Dataset missing: {DATA_CSV}")

    print("=" * 72)
    print("B1 smoke test: iSarcasmEval SubTask C (10 positive-surface sarcasm pairs)")
    print("Model: Qwen2.5-0.5B | Axes/scaler: E1 en/pole_50 | Forward-only")
    print("=" * 72)

    nrc = load_language_nrc("en", PROJECT_ROOT)
    pairs = load_isarcasm_task_c(DATA_CSV)
    pairs = annotate_surface_valence(pairs, nrc)
    selected = select_positive_surface_pairs(pairs, N_PAIRS)

    print(f"\nLoaded {len(pairs)} pairs from {DATA_CSV.name}")
    print(f"Selected {N_PAIRS} pairs where NRC surface valence(sarcastic) > surface valence(rephrase)\n")

    from src.model_loader import CausalModelLoader  # noqa: E402

    print("Loading Qwen2.5-0.5B (CPU)...")
    loader = CausalModelLoader("qwen", device="cpu")
    loader.load()
    projector, report = load_e1_projector(loader)
    print(
        f"E1 pole_50 scaler: valence scale={report['scaler']['valence']['scale']:.4f}, "
        f"bias={report['scaler']['valence']['bias']:.4f}\n"
    )

    rows = run_pair_batch(loader, projector, selected, aggregation="last")

    print(f"{'#':>2}  {'V_sarc':>7}  {'V_reph':>7}  {'Δ':>7}  {'NRC_s':>7}  {'NRC_r':>7}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['idx']:2d}  {r['valence_sarcastic']:7.3f}  {r['valence_rephrase']:7.3f}  "
            f"{r['delta']:7.3f}  {r['surface_valence_sarcastic']:7.3f}  "
            f"{r['surface_valence_rephrase']:7.3f}"
        )

    v_sarc = [r["valence_sarcastic"] for r in rows]
    v_reph = [r["valence_rephrase"] for r in rows]
    mean_s = sum(v_sarc) / len(v_sarc)
    mean_r = sum(v_reph) / len(v_reph)
    both_neg = sum(1 for s, r in zip(v_sarc, v_reph) if s < 0 and r < 0)
    close = sum(1 for s, r in zip(v_sarc, v_reph) if abs(s - r) < 0.15)

    print("\n" + "-" * 72)
    print(f"Mean projected valence — sarcastic: {mean_s:.3f}, rephrase: {mean_r:.3f}, Δ={mean_s - mean_r:.3f}")
    print(f"Both in negative zone (V<0): {both_neg}/{N_PAIRS}")
    print(f"|Δ|<0.15 (near overlap): {close}/{N_PAIRS}")

    print("\nSample pairs:")
    for r in rows[:3]:
        print(f"\n[{r['idx']}] SARC (V={r['valence_sarcastic']:.3f}, NRC={r['surface_valence_sarcastic']:.3f}):")
        print(f"    {r['sarcastic']}")
        print(f"    REPH (V={r['valence_rephrase']:.3f}, NRC={r['surface_valence_rephrase']:.3f}):")
        print(f"    {r['rephrase']}")

    print("\n" + "=" * 72)
    print("Smoke test complete (no files written).")
    print("=" * 72)


if __name__ == "__main__":
    main()
