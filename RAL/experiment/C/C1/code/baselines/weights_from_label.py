"""
Convert Mishra/Xpress label CSVs into constant 7-D FKCN-style weight vectors.

FKCN order (src/grimace_utils.py):
  neutral, happiness, surprise, fear, sadness, disgust, anger

Usage:
  python RAL/experiment/C/C1/code/baselines/weights_from_label.py \\
      --input RAL/experiment/C/C1/result/label_precompute/mishra_qwen.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FKCN_ORDER = [
    "neutral",
    "happiness",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
]

# Accept common aliases from Mishra / Xpress maps
ALIASES = {
    "happy": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    "happiness": "happiness",
    "sadness": "sadness",
}


def one_hot(name: str, intensity: float = 1.0) -> list[float]:
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in FKCN_ORDER:
        raise ValueError(f"Unknown emotion label: {name!r}")
    w = [0.0] * 7
    w[FKCN_ORDER.index(key)] = float(intensity)
    # Keep as a proper distribution if intensity < 1: put remainder on neutral
    if intensity < 1.0 - 1e-9 and key != "neutral":
        w[FKCN_ORDER.index("neutral")] = 1.0 - intensity
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="CSV with emotion_name column")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: <input_stem>_weights next to input",
    )
    ap.add_argument(
        "--intensity-col",
        default="",
        help="Optional column: low->0.5, high->1.0 (Xpress)",
    )
    args = ap.parse_args()

    outdir = args.outdir or args.input.with_name(args.input.stem + "_weights")
    outdir.mkdir(parents=True, exist_ok=True)

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # One file per stimulus id. Prefer last successful row (turn sentences: final segment).
    chosen: dict[str, dict] = {}
    for row in rows:
        sid = row["id"]
        name = (row.get("emotion_name") or "").strip()
        if not name or row.get("error"):
            continue
        chosen[sid] = row

    n_ok = 0
    for sid, row in chosen.items():
        name = row["emotion_name"].strip()
        intensity = 1.0
        if args.intensity_col and row.get(args.intensity_col):
            val = row[args.intensity_col].strip().lower()
            intensity = 0.5 if val == "low" else 1.0
        weights = one_hot(name, intensity)
        n_ok += 1
        out_path = outdir / f"{sid}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as wf:
            w = csv.writer(wf)
            w.writerow(FKCN_ORDER)
            w.writerow(weights)

    print(f"Wrote {n_ok} weight files to {outdir}")


if __name__ == "__main__":
    main()
