#!/usr/bin/env python3
"""
C1 Mirror 批分析：对每个 trial 的 video.mp4 + source_va.csv 调用 Py-Feat pipeline。

在装有 Py-Feat 的机器上运行（Windows 或 Linux，需先 setup mirror 环境）:

  # Windows (mirror 目录)
  cd RAL\\experiment\\C1\\code\\mirror
  setup_env.bat

  # 批处理
  python ..\\run_c1_mirror_batch.py \\
    --recordings ..\\..\\result\\recordings \\
    --time-offset-ms 150

输出写入各 trial 子目录 mirror/ 下（compare_metrics.csv 等）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent
MIRROR_DIR = CODE_DIR / "mirror"
PIPELINE = MIRROR_DIR / "video_va_pipeline.py"


def find_python(mirror_dir: Path) -> str:
    venv_win = mirror_dir / ".venv" / "Scripts" / "python.exe"
    venv_unix = mirror_dir / ".venv" / "bin" / "python"
    if venv_win.is_file():
        return str(venv_win)
    if venv_unix.is_file():
        return str(venv_unix)
    return sys.executable


def run_mirror_one(
    *,
    python: str,
    video: Path,
    source_va: Path,
    out_root: Path,
    time_offset_ms: float,
    device: str,
) -> Path:
    trial_mirror = out_root / "mirror"
    trial_mirror.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(PIPELINE),
        "--video", str(video),
        "--source-va", str(source_va),
        "--out-dir", str(trial_mirror),
        "--time-offset-ms", str(time_offset_ms),
        "--device", device,
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(MIRROR_DIR))
    results = sorted(trial_mirror.glob("*/compare_metrics.csv"))
    if results:
        return results[-1].parent
    nested = list(trial_mirror.glob("*/video_va.csv"))
    return nested[-1].parent if nested else trial_mirror


def main():
    parser = argparse.ArgumentParser(description="C1 batch mirror analysis")
    parser.add_argument(
        "--recordings",
        type=Path,
        default=C1_DIR / "result" / "recordings",
    )
    parser.add_argument("--time-offset-ms", type=float, default=150.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not PIPELINE.is_file():
        sys.exit(f"Mirror pipeline not found: {PIPELINE}")

    python = find_python(MIRROR_DIR)
    trials = sorted(p for p in args.recordings.iterdir() if p.is_dir())
    n = len(trials)
    print(f"Mirror python: {python}")
    print(f"Trials: {n}")
    print("=" * 50)

    done = 0
    for trial in trials:
        video = trial / "video.mp4"
        source_va = trial / "source_va.csv"
        if not video.is_file() or not source_va.is_file():
            print(f"[skip] {trial.name}: missing video or source_va")
            continue
        existing = list(trial.glob("mirror/**/compare_metrics.csv"))
        if args.skip_existing and existing:
            print(f"[skip] {trial.name}: already analyzed")
            continue
        print(f"\n[C1 Mirror] {done + 1}/{n}  Analyzing {trial.name} …")
        run_mirror_one(
            python=python,
            video=video,
            source_va=source_va,
            out_root=trial,
            time_offset_ms=args.time_offset_ms,
            device=args.device,
        )
        done += 1
        print(f"[C1 Mirror] {done}/{n}  完成 {trial.name}")

    print(f"\n[C1 Mirror] 全部完成 {done}/{n}")


if __name__ == "__main__":
    main()
