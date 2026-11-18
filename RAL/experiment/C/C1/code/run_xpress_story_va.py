#!/usr/bin/env python3
"""Py-Feat VA for the Xpress storytelling re-record, then the C1 compare.

Detector settings match the July field run (summary.json):
  frame_step=1, face_threshold=0.7, smooth_ms=200, interpolate_limit_ms=250.
Comparison matches run_c1_multicond_compare.py:
  command = ours/source_va, fixed lag 0.5 s, smooth tau 0.6 s,
  calibration fit on the same mishra/xpress still faces.
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
MIRROR = CODE / "mirror"
C1 = CODE.parents[0]
STORY = C1 / "result/xpress_story_20261111_2k"
FIELD = C1 / "result/multicond_noanchor_20260731"
OUT = C1 / "output/xpress_story_20261111_2k"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(MIRROR))

import video_va_pipeline as pipe  # noqa: E402
from run_c1_multicond_compare import (  # noqa: E402
    analyze_system,
    attach_command,
    fit_static_calibration,
    load_face,
)


def detect_one(detector, video: Path, out_dir: Path) -> Path:
    info = pipe.read_video_info(video)
    tmp = Path(tempfile.mkdtemp(prefix="c1_frames_"))
    try:
        frames = pipe._extract_frames_opencv(video, 1, tmp)
        print(f"  detect {len(frames)} frames {info.width}x{info.height}", flush=True)
        output = detector.detect(
            [str(p) for p in frames],
            data_type="image",
            batch_size=pipe.default_batch_size(detector.device),
            face_detection_threshold=0.5,
            num_workers=0,
            progress_bar=False,
        )
        raw = pipe.dataframe_from_detector_output(output)
        if "input" in raw.columns or "FileName" in raw.columns:
            col = "input" if "input" in raw.columns else "FileName"
            raw["frame"] = [
                int(Path(str(name)).stem.split("_")[-1]) for name in raw[col].astype(str)
            ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    run_dir = pipe.create_run_dir(out_dir, video)
    raw.to_csv(run_dir / "video_va_full.csv", index=False, encoding="utf-8-sig")
    timeline = pipe.build_frame_timeline(
        raw,
        info,
        face_threshold=0.7,
        smooth_ms=200,
        interpolate_limit_ms=250,
    )
    timeline.to_csv(run_dir / "video_va.csv", index=False, encoding="utf-8-sig")
    valid = int(timeline["face_valid"].sum()) if "face_valid" in timeline.columns else -1
    print(f"  saved {run_dir.name} valid={valid}/{len(timeline)}", flush=True)
    summary = {
        "video": {
            "path": str(video),
            "fps": info.fps,
            "total_frames": info.total_frames,
            "width": info.width,
            "height": info.height,
            "duration_s": info.duration_s,
        },
        "device": detector.device,
        "batch_size": pipe.default_batch_size(detector.device),
        "frame_step": 1,
        "face_threshold": 0.7,
        "smooth_ms": 200,
        "interpolate_limit_ms": 250,
        "valid_rows": valid,
        "total_output_rows": int(len(timeline)),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--story", type=Path, default=STORY)
    ap.add_argument("--field", type=Path, default=FIELD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()
    story, field, out = args.story, args.field, args.out

    pipe.ensure_runtime_dependencies()
    out.mkdir(parents=True, exist_ok=True)
    device = pipe.choose_device("auto")
    print(f"device={device}", flush=True)
    detector = pipe.make_detector(device)
    detector.device = device

    for trial in sorted(p for p in story.iterdir() if (p / "video.mp4").is_file()):
        mirror = trial / "mirror"
        if list(mirror.glob("*/video_va.csv")):
            print(f"[{trial.name}] mirror exists", flush=True)
            continue
        print(f"[{trial.name}]", flush=True)
        detect_one(detector, trial / "video.mp4", mirror)

    if args.detect_only:
        print("detect-only done", flush=True)
        return

    calib = fit_static_calibration(field)
    print(f"[calib] V={np.round(calib[0], 3)} A={np.round(calib[1], 3)}", flush=True)
    trials = []
    for tid_dir in sorted((field / "ours").iterdir()):
        if not tid_dir.is_dir():
            continue
        src_path = tid_dir / "source_va.csv"
        if not src_path.is_file() or len(pd.read_csv(src_path)) < 5:
            continue
        face = load_face(story / tid_dir.name, 0.6, calib)
        if face is None:
            print(f"[skip] {tid_dir.name}", flush=True)
            continue
        trials.append(attach_command(face, pd.read_csv(src_path), 0.6, True))

    result = analyze_system(trials, window=2.0, n_perm=500, seed=0, fixed_lag=0.5)
    (out / "xpress_story_va.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    v = result["event_2afc_clustered"]["valence"]["trial_mean_accuracy"]
    a = result["event_2afc_clustered"]["arousal"]["trial_mean_accuracy"]
    rv = result["variance_decomposition"]["valence"]["R2_gain"]
    ra = result["variance_decomposition"]["arousal"]["R2_gain"]
    row = {
        "system": "xpress_story",
        "n_trials": result["n_trials"],
        "V_direction_agreement_pct": round(100 * v, 1),
        "A_direction_agreement_pct": round(100 * a, 1),
        "V_R2_gain_pp": round(100 * rv, 1),
        "A_R2_gain_pp": round(100 * ra, 1),
    }
    (out / "xpress_story_table.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
