#!/usr/bin/env python3
"""Replay C1 source_va trajectories onto Grimace and write MP4 videos.

Pipeline (aligned with Sec. III-C / hyperparam ablation):
  source_va VA → clip → FKCN(P=13) → EMA(α=0.6)
  → 10 fps smoothstep + rate limit → Grimace 6-D → Playwright frames → MP4

Usage (repo root):
  python RAL/experiment/C/C1/code/render_grimace_from_source_va.py \\
      --ids 013 --conditions ours shuffled --out-dir /tmp/grimace_smoke
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from src.face_interpolator import FaceFrameInterpolator  # noqa: E402
from src.fkcn import FKCN  # noqa: E402
from src.grimace_renderer import GrimaceFrameRenderer  # noqa: E402
from src.grimace_utils import fkcn_to_grimace_weights, smooth_weights_batch  # noqa: E402

from run_c1_hyperparam_ablation import (  # noqa: E402
    FPS,
    MAX_DELTA,
    SETTLE_FRAMES,
    TRANSITION_MS,
    replay_display_path,
)

DEFAULT_FIELD = (
    CODE_DIR.parent / "result" / "multicond_noanchor_20260731" / "ours"
)


def load_source(path: Path) -> dict:
    src = pd.read_csv(path)
    st = src["timestamp_s"].to_numpy(float)
    v_col = "raw_valence" if "raw_valence" in src.columns else "valence"
    a_col = "raw_arousal" if "raw_arousal" in src.columns else "arousal"
    v = np.clip(src[v_col].to_numpy(float), -1.0, 1.0)
    a = np.clip(src[a_col].to_numpy(float), -1.0, 1.0)
    tokens = src["token"].astype(str).tolist() if "token" in src.columns else [""] * len(st)
    return dict(st=st, v=v, a=a, tokens=tokens)


def shuffle_trial(tr: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr["v"]))
    return dict(
        st=tr["st"],  # keep original timing so duration matches
        v=tr["v"][perm],
        a=tr["a"][perm],
        tokens=[tr["tokens"][i] for i in perm],
        perm=perm.tolist(),
    )


def build_frame_weights(
    tr: dict, alpha: float, lead_s: float, trail_s: float, realtime: bool = False,
) -> tuple[np.ndarray, list[list[float]], list[list[float]]]:
    """Return (frame_times, grimace_6d_per_frame, source_va_rows for export)."""
    fkcn = FKCN(device="cpu")
    points = list(zip(tr["v"].tolist(), tr["a"].tolist()))
    raw_w = fkcn.compute_weights_batch(points)
    smooth_w = smooth_weights_batch(raw_w, alpha=alpha)
    src_times: list[float] = []
    ft, frames_7, _sat = replay_display_path(tr["st"], smooth_w, src_times=src_times)
    grimace = fkcn_to_grimace_weights(frames_7.tolist())

    # Hold each face until the next token so video time matches the token clock.
    if realtime and grimace:
        t0 = float(tr["st"][0])
        t1 = float(tr["st"][-1]) + trail_s
        n = int(round((t1 - t0) * FPS)) + 1
        expanded: list[list[float]] = []
        k = 0
        for i in range(n):
            t = t0 + i / FPS
            while k + 1 < len(src_times) and src_times[k + 1] <= t + 1e-6:
                k += 1
            expanded.append(grimace[min(k, len(grimace) - 1)])
        ft = np.arange(n) / FPS
        return ft, expanded, smooth_w

    # Pad lead/trail with first/last face so Py-Feat has settle frames.
    if lead_s > 0 and len(grimace):
        n_lead = max(1, int(round(lead_s * FPS)))
        ft = np.concatenate([ft[0] - np.arange(n_lead, 0, -1) / FPS, ft])
        grimace = [grimace[0]] * n_lead + grimace
    if trail_s > 0 and len(grimace):
        n_trail = max(1, int(round(trail_s * FPS)))
        ft = np.concatenate([ft, ft[-1] + np.arange(1, n_trail + 1) / FPS])
        grimace = grimace + [grimace[-1]] * n_trail

    # Shift times so video starts at 0.
    ft = ft - float(ft[0])
    return ft, grimace, smooth_w


def write_source_va_export(
    path: Path,
    tr: dict,
    condition: str,
    seed: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["timestamp_s", "valence", "arousal", "token", "raw_valence", "raw_arousal"]
        if condition == "shuffled":
            header.append("orig_token_idx")
        w.writerow(header)
        for i in range(len(tr["st"])):
            row = [
                f"{tr['st'][i]:.4f}",
                f"{tr['v'][i]:.6f}",
                f"{tr['a'][i]:.6f}",
                tr["tokens"][i],
                f"{tr['v'][i]:.6f}",
                f"{tr['a'][i]:.6f}",
            ]
            if condition == "shuffled":
                row.append(tr["perm"][i])
            w.writerow(row)


def frames_to_mp4(png_bytes_list: list[bytes], out_mp4: Path, fps: float) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grimace_frames_") as tmp:
        tmp_dir = Path(tmp)
        for i, png in enumerate(png_bytes_list):
            (tmp_dir / f"frame_{i:05d}.png").write_bytes(png)
        # White-background Grimace; yuv420p for Windows/OpenCV compatibility.
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def render_condition(
    trial_id: str,
    condition: str,
    field_root: Path,
    out_dir: Path,
    alpha: float,
    seed: int,
    width: int,
    height: int,
    lead_s: float,
    trail_s: float,
    realtime: bool = False,
) -> Path:
    src_path = field_root / trial_id / "source_va.csv"
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    tr = load_source(src_path)
    if condition == "shuffled":
        tr = shuffle_trial(tr, seed=seed + int(trial_id))
    elif condition != "ours":
        raise ValueError(f"unsupported condition: {condition}")

    ft, grimace, _ = build_frame_weights(tr, alpha, lead_s, trail_s, realtime=realtime)
    print(
        f"[{trial_id}/{condition}] tokens={len(tr['v'])} frames={len(grimace)} "
        f"duration={ft[-1]:.2f}s",
        flush=True,
    )

    meta_dir = out_dir / "meta" / condition
    write_source_va_export(
        meta_dir / f"{trial_id}_source_va.csv",
        tr,
        condition,
        seed if condition == "shuffled" else None,
    )
    # Also write a session stub for later analysis.
    (meta_dir / f"{trial_id}_session.json").write_text(
        '{"id":"%s","condition":"%s","carrier":"grimace","ema_alpha":%s,"fps":%s}\n'
        % (trial_id, condition, alpha, FPS),
        encoding="utf-8",
    )

    pngs: list[bytes] = []
    with GrimaceFrameRenderer(width=width, height=height) as renderer:
        # Neutral warm-up so the first expression isn't a cold start glitch.
        pngs.append(renderer.render([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        for i, w6 in enumerate(grimace):
            pngs.append(renderer.render(list(w6)))
            if (i + 1) % 20 == 0 or i + 1 == len(grimace):
                print(f"  rendered {i + 1}/{len(grimace)}", flush=True)

    # Drop the warm-up frame from the video timeline (keep only grimace frames).
    pngs = pngs[1:]
    out_mp4 = out_dir / "videos" / condition / f"{trial_id}.mp4"
    frames_to_mp4(pngs, out_mp4, FPS)
    print(f"  wrote {out_mp4}", flush=True)
    return out_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-root", type=Path, default=DEFAULT_FIELD)
    ap.add_argument("--ids", nargs="+", default=["013"])
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["ours", "shuffled"],
        choices=["ours", "shuffled"],
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=CODE_DIR.parent / "output" / "grimace_smoke",
    )
    ap.add_argument("--ema-alpha", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--lead-s", type=float, default=1.0)
    ap.add_argument("--trail-s", type=float, default=1.0)
    ap.add_argument("--realtime", action="store_true",
                    help="hold each face until the next token so video time matches the token clock")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tid in args.ids:
        for cond in args.conditions:
            render_condition(
                tid,
                cond,
                args.field_root,
                args.out_dir,
                args.ema_alpha,
                args.seed,
                args.width,
                args.height,
                args.lead_s,
                args.trail_s,
                args.realtime,
            )
    print(f"\nDone. Videos under {args.out_dir / 'videos'}")


if __name__ == "__main__":
    main()
