#!/usr/bin/env python3
"""汇总各 trial 的 compare_metrics.csv → result/c1_summary.json + output/c1_table.md"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent


def aggregate_metrics(result_root: Path) -> List[dict]:
    rows = []
    for trial_dir in sorted(result_root.iterdir()):
        if not trial_dir.is_dir():
            continue
        metrics_files = list(trial_dir.glob("mirror/**/compare_metrics.csv"))
        if not metrics_files:
            continue
        with metrics_files[0].open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({"trial_id": trial_dir.name, **row})
    return rows


def write_summary(rows: List[dict], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "n_trials": len({r["trial_id"] for r in rows}),
        "metrics": {},
    }
    for metric in ("valence", "arousal", "va_distance"):
        sub = [r for r in rows if r.get("metric") == metric and r.get("n", "0") != "0"]
        if not sub:
            continue

        def mean_key(key: str) -> float:
            vals = [float(r[key]) for r in sub if r.get(key) not in (None, "", "nan")]
            return statistics.mean(vals) if vals else float("nan")

        summary["metrics"][metric] = {
            "pearson_r_mean": mean_key("pearson_r"),
            "mae_mean": mean_key("mae"),
            "rmse_mean": mean_key("rmse"),
            "ccc_mean": mean_key("ccc"),
            "n_trials": len(sub),
        }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# C1 Mirror Test 汇总\n", f"有效 trial 数: {summary['n_trials']}\n"]
    for metric, stats in summary.get("metrics", {}).items():
        lines.append(f"## {metric}\n")
        lines.append(f"- Pearson r (mean): {stats['pearson_r_mean']:.4f}\n")
        lines.append(f"- MAE (mean): {stats['mae_mean']:.4f}\n")
        lines.append(f"- RMSE (mean): {stats['rmse_mean']:.4f}\n")
        lines.append(f"- CCC (mean): {stats['ccc_mean']:.4f}\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, default=C1_DIR / "result" / "recordings")
    parser.add_argument("--out-json", type=Path, default=C1_DIR / "result" / "c1_summary.json")
    parser.add_argument("--out-md", type=Path, default=C1_DIR / "output" / "c1_table.md")
    args = parser.parse_args()

    rows = aggregate_metrics(args.recordings)
    if not rows:
        print("No compare_metrics.csv found. Run run_c1_mirror_batch.py first.")
        return

    per_trial = args.recordings / "c1_all_metrics.csv"
    with per_trial.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_summary(rows, args.out_json, args.out_md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {per_trial}")


if __name__ == "__main__":
    main()
