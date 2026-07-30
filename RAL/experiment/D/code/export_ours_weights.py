"""
Export EmotionManifold (ours) per-token 7-D FKCN weights for C1 stimuli.

Pipeline (aligned with B1 / paper axes):
  Qwen2.5-0.5B CausalLM hidden states
  → E1 en/pole_50 EmotionProjector (+ scaler)
  → FKCN(V, A) → EMA(α=0.6)

Usage (repo root, prefer conda python):
  python RAL/experiment/D/code/export_ours_weights.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

B1_CODE = ROOT / "RAL/experiment/B/B1/code"
if str(B1_CODE) not in sys.path:
    sys.path.insert(0, str(B1_CODE))

STIMULI = ROOT / "RAL/experiment/C/C1/data/stimuli.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "precompute"
WEIGHTS_DIR = OUT_DIR / "ours_weights"
LATENCY_CSV = OUT_DIR / "ours_latency.csv"

FKCN_ORDER = [
    "neutral",
    "happiness",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
]


def load_stimuli(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--ema-alpha", type=float, default=0.6)
    args = ap.parse_args()

    from src.model_loader import CausalModelLoader
    from src.fkcn import FKCN
    from src.grimace_utils import smooth_weights_batch
    from b1_core import load_e1_projector

    print(f"Loading Qwen CausalLM on {args.device}...", flush=True)
    loader = CausalModelLoader("qwen", device=args.device)
    loader.load()
    projector, report = load_e1_projector(loader)
    fkcn = FKCN(device="cpu")
    print(
        f"E1 pole_50 scaler: V={report['scaler']['valence']['scale']:.4f}, "
        f"A={report['scaler']['arousal']['scale']:.4f}",
        flush=True,
    )

    stimuli = load_stimuli(args.stimuli)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    latency_rows = []

    for row in stimuli:
        sid = row["id"]
        text = row["user_prompt"]
        t0 = time.perf_counter()
        coords, tokens = projector.project_text(loader, text, apply_scaling=True)
        valence = coords.get("valence", [])
        arousal = coords.get("arousal", [])
        dominance = coords.get("dominance", [0.0] * len(valence))
        points = list(zip(valence, arousal))
        raw_w = fkcn.compute_weights_batch(points)
        smooth_w = smooth_weights_batch(raw_w, alpha=args.ema_alpha)
        sentence_ms = (time.perf_counter() - t0) * 1000
        n = len(tokens)
        mean_tok_ms = sentence_ms / n if n else float("nan")

        out_path = WEIGHTS_DIR / f"{sid}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["token_idx", "token", "valence", "arousal", "dominance"] + FKCN_ORDER
            )
            for i, tok in enumerate(tokens):
                w.writerow(
                    [
                        i,
                        tok,
                        f"{valence[i]:.6f}",
                        f"{arousal[i]:.6f}",
                        f"{dominance[i]:.6f}" if i < len(dominance) else "",
                        *[f"{x:.6f}" for x in smooth_w[i]],
                    ]
                )

        latency_rows.append(
            {
                "id": sid,
                "category": row.get("category", ""),
                "n_tokens": n,
                "sentence_ms": f"{sentence_ms:.2f}",
                "mean_token_ms": f"{mean_tok_ms:.2f}",
                "user_prompt": text,
            }
        )
        print(
            f"[{sid}] n={n} sentence={sentence_ms:.1f}ms "
            f"mean_tok={mean_tok_ms:.2f}ms → {out_path.name}",
            flush=True,
        )

    with LATENCY_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = ["id", "category", "n_tokens", "sentence_ms", "mean_token_ms", "user_prompt"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(latency_rows)
    print(f"Wrote {len(latency_rows)} weight files to {WEIGHTS_DIR}")
    print(f"Wrote {LATENCY_CSV}")


if __name__ == "__main__":
    main()
