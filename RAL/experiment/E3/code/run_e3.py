#!/usr/bin/env python3
"""
E3：EN-only / ZH-only pole 构造下的 VAD 轴正交性、跨语聚类 vs 随机 3D 基线。

用法（仓库根目录）:
  python RAL/experiment/E3/code/run_e3.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
E3_DIR = CODE_DIR.parent
E1_CODE = E3_DIR.parent / "E1" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(E1_CODE))

from project_root import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from e1_core import extract_axes_from_embedding  # noqa: E402
from e3_core import (  # noqa: E402
    evaluate_clusters,
    load_json,
    orthogonality,
    random_baseline_clusters,
    summarize_condition,
)

N_RANDOM_TRIALS = 100
RANDOM_SEED = 42


def _load_model():
    from src.model_loader import CausalModelLoader

    loader = CausalModelLoader("qwen", device="cpu")
    loader.load()
    return loader


def _scaler_params_from_report(report: dict) -> dict:
    return {d: dict(report["scaler"][d]) for d in ("valence", "arousal", "dominance")}


def _run_condition_from_e1(
    name: str,
    pole_lexicon_path: Path,
    report_path: Path,
    emb,
    tokenizer,
    clusters: list,
) -> dict:
    report = load_json(report_path)
    axes = extract_axes_from_embedding(emb, tokenizer, str(pole_lexicon_path), device="cpu")
    scaler_params = _scaler_params_from_report(report)
    orth = orthogonality(axes)
    cluster_cos, skipped = evaluate_clusters(
        emb, tokenizer, clusters, axes, scaler_params
    )
    return summarize_condition(
        name,
        str(pole_lexicon_path.relative_to(PROJECT_ROOT)),
        scaler_params,
        orth,
        cluster_cos,
        skipped,
    )


def _write_table_tex(path: Path, conditions: dict, baseline_mean: float) -> None:
    rows = [("EN-only", "en_only"), ("ZH-only", "zh_only")]
    lines = [
        "% E3 table (IEEE/RAL style, booktabs)",
        "\\begin{table}[t]",
        "\\caption{Orthogonality of mined VAD axes and cross-lingual emotion-cluster "
        "cohesion under two pole-lexicon constructions ($N{=}50$ per pole, "
        "Qwen2.5-0.5B). Cluster cohesion is the mean intra-cluster pairwise cosine "
        "over five predefined multilingual emotion sets (five languages each). "
        f"The random baseline is shared across rows ({N_RANDOM_TRIALS} trials per "
        "cluster, no scaler); value shown in parentheses.}",
        "\\label{tab:e3_axis_cluster}",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Pole lexicon & $|\\cos(V,A)|$ & $|\\cos(A,D)|$ & $|\\cos(V,D)|$ "
        "& Cluster cos. (rand.) \\\\",
        "\\midrule",
    ]
    for label, key in rows:
        c = conditions[key]
        o = c["orthogonality"]
        cc = c["cluster_cos"]["mean"]
        lines.append(
            f"{label} & {o['va']:.3f} & {o['ad']:.3f} & {o['vd']:.3f} "
            f"& {cc:.3f} ({baseline_mean:.3f}) \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_description_txt(path: Path, metrics: dict) -> None:
    b = metrics["random_baseline_cluster_cos"]["mean"]
    en = metrics["conditions"]["en_only"]
    zh = metrics["conditions"]["zh_only"]
    text = f"""E3 Experiment: VAD Axis Geometry and Cross-Lingual Emotion Clustering
================================================================================

Purpose
-------
Compare English-only vs Chinese-only pole lexicons (N=50 words per pole, six poles)
on: (1) axis orthogonality; (2) cross-lingual emotion-cluster cohesion vs a
shared random 3-D baseline.

Model and projection
--------------------
- Model: Qwen2.5-0.5B (same as E1)
- Axes: axis_d = normalize(mean(pos_pole) - mean(neg_pole)) for V/A/D
- Scaled coordinate: raw = normalize(emb) · axis; score = raw × scale + bias
- Random baseline (shared): per cluster, 100 trials; one random 3-D orthonormal
  basis per trial; no AdaptiveScaler. Same value for both rows.

Conditions
----------
1. EN-only: E1 English B1 pole_50 + scaler from E1 report.
2. ZH-only: E1 Chinese B1 pole_50 + scaler from E1 report.

Emotion clusters (5 × 5 languages, all single-token)
------------------------------------------------------
terror, comfort, boss, strength, zen — see E3/data/clusters.json.

Results summary
---------------
                    |cos(V,A)| |cos(A,D)| |cos(V,D)| cluster cos (rand.)
EN-only             {en['orthogonality']['va']:.3f}      {en['orthogonality']['ad']:.3f}      {en['orthogonality']['vd']:.3f}      {en['cluster_cos']['mean']:.3f} ({b:.3f})
ZH-only             {zh['orthogonality']['va']:.3f}      {zh['orthogonality']['ad']:.3f}      {zh['orthogonality']['vd']:.3f}      {zh['cluster_cos']['mean']:.3f} ({b:.3f})

Interpretation
--------------
Both monolingual pole constructions yield cluster cohesion well above the shared
random baseline, supporting that mined VAD subspaces capture cross-lingual
emotion structure under either English or Chinese pole anchors.
"""
    path.write_text(text, encoding="utf-8")


def main():
    t0 = time.time()
    data_dir = E3_DIR / "data"
    result_dir = E3_DIR / "result"
    output_dir = E3_DIR / "output"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters_data = load_json(data_dir / "clusters.json")
    clusters = clusters_data["clusters"]

    e1_result = E3_DIR.parent / "E1" / "result"
    en_pole = e1_result / "en" / "pole_50"
    zh_pole = e1_result / "zh" / "pole_50"

    print("加载 Qwen2.5-0.5B...")
    loader = _load_model()
    emb = loader.get_output_embedding_matrix()
    tokenizer = loader.tokenizer

    print("随机 3D 基线（全条件共享）...")
    baseline = random_baseline_clusters(
        emb, tokenizer, clusters, n_trials=N_RANDOM_TRIALS, seed=RANDOM_SEED
    )
    print(f"  baseline mean cluster cos = {baseline['mean']:.3f}")

    print("EN-only...")
    en_cond = _run_condition_from_e1(
        "en_only",
        en_pole / "pole_lexicon.json",
        en_pole / "report.json",
        emb,
        tokenizer,
        clusters,
    )
    print(f"  cluster mean = {en_cond['cluster_cos']['mean']:.3f}")

    print("ZH-only...")
    zh_cond = _run_condition_from_e1(
        "zh_only",
        zh_pole / "pole_lexicon.json",
        zh_pole / "report.json",
        emb,
        tokenizer,
        clusters,
    )
    print(f"  cluster mean = {zh_cond['cluster_cos']['mean']:.3f}")

    metrics = {
        "experiment": "E3",
        "model": "Qwen2.5-0.5B",
        "pole_n": 50,
        "n_random_trials": N_RANDOM_TRIALS,
        "random_seed": RANDOM_SEED,
        "random_baseline_cluster_cos": baseline,
        "conditions": {
            "en_only": en_cond,
            "zh_only": zh_cond,
        },
    }

    metrics_path = result_dir / "e3_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _write_table_tex(output_dir / "e3_table.tex", metrics["conditions"], baseline["mean"])
    _write_description_txt(output_dir / "e3_description.txt", metrics)

    print(f"\n完成 ({time.time() - t0:.1f}s)")
    print(f"  metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"  table:   {(output_dir / 'e3_table.tex').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
