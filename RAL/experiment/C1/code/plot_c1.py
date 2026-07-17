#!/usr/bin/env python3
"""C1 汇总图：Pearson/MAE 柱状图 + token-frame 对齐散点。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recordings",
        type=Path,
        default=C1_DIR / "result" / "recordings",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=C1_DIR / "output",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 汇总柱状图
    summary_path = C1_DIR / "result" / "c1_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        names = list(metrics.keys())
        pearson = [metrics[m].get("pearson_r_mean", 0) for m in names]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.bar(names, pearson, color=["#4C72B0", "#55A868", "#C44E52"][: len(names)])
        ax.set_ylabel("Pearson r (mean)")
        ax.set_title("C1 Mirror Test")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout()
        fig.savefig(args.out_dir / "c1_pearson_bar.png")
        plt.close(fig)

    # 逐 trial 对齐散点（取 compare_aligned.csv）
    aligned_files = sorted(args.recordings.glob("*/mirror/*/compare_aligned.csv"))
    if aligned_files:
        import pandas as pd

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        all_v, all_a = [], []
        for p in aligned_files:
            df = pd.read_csv(p)
            all_v.extend(df["source_valence"].dropna().tolist())
            all_v.extend(df["detected_valence"].dropna().tolist())
            axes[0].scatter(
                df["source_valence"], df["detected_valence"],
                s=8, alpha=0.3,
            )
            axes[1].scatter(
                df["source_arousal"], df["detected_arousal"],
                s=8, alpha=0.3,
            )
        for ax, dim in zip(axes, ("Valence", "Arousal")):
            ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlabel(f"Source {dim}")
            ax.set_ylabel(f"Detected {dim}")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
        fig.suptitle("C1 Token-aligned vs Frame-detected VA")
        fig.tight_layout()
        fig.savefig(args.out_dir / "c1_scatter_va.png")
        plt.close(fig)

    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
