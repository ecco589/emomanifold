#!/usr/bin/env python3
"""
E2：跨模型 Spearman 对比（NIPS Table 1 复现 + Qwen2.5-0.5B）。

固定 pole_n=50、英文 B1 设置；五模型数值来自 NIPS B1_cross_model_spearman，
Qwen2.5-0.5B 来自 E1 `result/en/pole_50/report.json`（同一交集 hold-out）。

用法（仓库根目录）:
  python RAL/experiment/E2/code/run_e2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
E2_DIR = CODE_DIR.parent
E1_DIR = E2_DIR.parent / "E1"
sys.path.insert(0, str(CODE_DIR))

from project_root import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root()
POLE_N = 50
DIMS = ("valence", "arousal", "dominance")

NIPS_MODELS: list[tuple[str, str]] = [
    ("gemma22_2b", "Gemma-2.2-2B"),
    ("llama32_3b", "Llama-3.2-3B"),
    ("mistral7b_v03", "Mistral-7B-v0.3"),
    ("qwen25_7b", "Qwen2.5-7B"),
    ("llama31_8b", "Llama-3.1-8B"),
]
QWEN_05B_KEY = "qwen25_0_5b"
QWEN_05B_LABEL = "Qwen2.5-0.5B"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _round_rho(v: float) -> float:
    return round(float(v), 3)


def _mean_rho(row: dict) -> float:
    return round(sum(row[d] for d in DIMS) / 3, 3)


def build_table_rows() -> tuple[list[dict], dict]:
    nips_path = E1_DIR / "data" / "nips_b1_per_model_spearman.json"
    qwen_report = E1_DIR / "result" / "en" / "pole_50" / "report.json"

    nips = _load_json(nips_path)
    pole_key = str(POLE_N)
    if pole_key not in nips["by_pole"]:
        raise KeyError(f"pole_n={POLE_N} not in {nips_path}")

    qwen = _load_json(qwen_report)
    qwen_rho = {
        d: _round_rho(qwen["spearman"][d]["rho"]) for d in DIMS
    }

    rows: list[dict] = []
    for key, label in NIPS_MODELS:
        raw = nips["by_pole"][pole_key][key]
        rho = {d: _round_rho(raw[d]) for d in DIMS}
        rows.append({
            "model_key": key,
            "model": label,
            "source": "NIPS B1_cross_model_spearman",
            "pole_n": POLE_N,
            "rho": rho,
            "mean_rho": _mean_rho(rho),
        })

    rows.append({
        "model_key": QWEN_05B_KEY,
        "model": QWEN_05B_LABEL,
        "source": "E1 en/pole_50/report.json",
        "pole_n": POLE_N,
        "holdout": "nips_holdout_intersection_en.json",
        "rho": qwen_rho,
        "mean_rho": _mean_rho(qwen_rho),
    })

    meta = {
        "experiment": "E2",
        "description": (
            "Cross-model Spearman at pole_n=50 (NIPS Table 1 style). "
            "NIPS five models: original B1 hold-out per model. "
            "Qwen2.5-0.5B: five-model tokenizer-intersection hold-out (E1)."
        ),
        "pole_n": POLE_N,
        "nips_source": str(nips_path.relative_to(PROJECT_ROOT)),
        "qwen05b_source": str(qwen_report.relative_to(PROJECT_ROOT)),
        "dimensions": list(DIMS),
    }
    return rows, meta


def _write_table_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "% E2 table (IEEE/RAL style, booktabs; NIPS Table 1 + Qwen2.5-0.5B)",
        "\\begin{table}[t]",
        "\\caption{Cross-model Spearman $\\rho$ on English NRC-VAD hold-out words "
        "($N_{\\mathrm{pole}}{=}50$ per axis pole). Rows 1--5 reproduce NIPS B1 "
        "cross-model results; row 6 adds Qwen2.5-0.5B (this work).}",
        "\\label{tab:e2_cross_model_spearman}",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & $\\rho_V$ & $\\rho_A$ & $\\rho_D$ \\\\",
        "\\midrule",
    ]
    for i, row in enumerate(rows):
        r = row["rho"]
        label = row["model"]
        if i == len(rows) - 1:
            label = f"{label} (ours)"
        lines.append(
            f"{label} & {r['valence']:.3f} & {r['arousal']:.3f} & {r['dominance']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_md(path: Path, rows: list[dict]) -> None:
    lines = [
        "# E2 Cross-model Spearman (pole_n=50)",
        "",
        "| Model | ρ_V | ρ_A | ρ_D | mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        r = row["rho"]
        suffix = " **(ours)**" if row["model_key"] == QWEN_05B_KEY else ""
        lines.append(
            f"| {row['model']}{suffix} | {r['valence']:.3f} | {r['arousal']:.3f} "
            f"| {r['dominance']:.3f} | {row['mean_rho']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_description_txt(path: Path, rows: list[dict], meta: dict) -> None:
    body = "\n".join(
        f"{row['model']:18s}  V={row['rho']['valence']:.3f}  "
        f"A={row['rho']['arousal']:.3f}  D={row['rho']['dominance']:.3f}  "
        f"mean={row['mean_rho']:.3f}"
        for row in rows
    )
    text = f"""E2 Experiment: Cross-Model Spearman Generalization (NIPS Table 1 + Qwen 0.5B)
==================================================================================

Purpose
-------
Show that the emotion-axis mining + projection pipeline generalizes across LLM
backbones, following NIPS B1 cross-model Spearman (Table 1) and adding our
deployment model Qwen2.5-0.5B.

Setup
-----
- pole_n = 50 per axis pole (six poles); English NRC-VAD bipolar hold-out
- Rows 1--5: NIPS five-model B1 results (source: {meta['nips_source']})
- Row 6: Qwen2.5-0.5B from E1 en/pole_50 (five-model tokenizer-intersection hold-out)

Results (Spearman rho)
----------------------
{body}

Note
----
NIPS models use each paper's original B1 hold-out protocol; Qwen2.5-0.5B uses the
reformed intersection hold-out aligned with E1. Direct numeric comparison is
indicative; the key claim is that 0.5B remains in the same range as larger models.
"""
    path.write_text(text, encoding="utf-8")


def main():
    data_dir = E2_DIR / "data"
    result_dir = E2_DIR / "result"
    output_dir = E2_DIR / "output"
    for d in (data_dir, result_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, meta = build_table_rows()
    payload = {**meta, "rows": rows}

    data_path = data_dir / "table1_pole50.json"
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    result_path = result_dir / "e2_metrics.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    _write_table_tex(output_dir / "e2_table.tex", rows)
    _write_table_md(output_dir / "e2_spearman_table.md", rows)
    _write_description_txt(output_dir / "e2_description.txt", rows, meta)

    print(f"Wrote {data_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {output_dir / 'e2_table.tex'}")
    for row in rows:
        print(
            f"  {row['model']:18s}  "
            f"V={row['rho']['valence']:.3f}  A={row['rho']['arousal']:.3f}  "
            f"D={row['rho']['dominance']:.3f}"
        )


if __name__ == "__main__":
    main()
