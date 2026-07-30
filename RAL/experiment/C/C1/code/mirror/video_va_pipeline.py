from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_FACE_THRESHOLD = 0.70
DEFAULT_SMOOTH_MS = 200
DEFAULT_INTERPOLATE_LIMIT_MS = 250

cv2 = None
np = None
pd = None


@dataclass
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float


@dataclass
class RunSummary:
    video: VideoInfo
    device: str
    batch_size: int
    frame_step: int
    face_threshold: float
    smooth_ms: int
    interpolate_limit_ms: int
    total_output_rows: int
    detected_rows: int
    valid_rows: int
    valid_ratio: float
    output_dir: str
    created_at: str


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def ensure_runtime_dependencies() -> None:
    global cv2, np, pd

    if cv2 is not None and np is not None and pd is not None:
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
        import pandas as pd_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc.name}. Run setup_env.bat first."
        ) from exc

    cv2 = cv2_module
    np = np_module
    pd = pd_module


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    candidate_names = [normalize_name(name) for name in candidates]
    normalized = {normalize_name(column): str(column) for column in df.columns}

    for candidate in candidate_names:
        if candidate in normalized:
            return normalized[candidate]

    for candidate in candidate_names:
        for norm, original in normalized.items():
            if candidate and candidate in norm:
                return original

    if required:
        raise KeyError(
            "Could not find a required column. Tried: "
            + ", ".join(candidates)
            + "\nAvailable columns: "
            + ", ".join(map(str, df.columns))
        )

    return None


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def default_batch_size(device: str) -> int:
    if device == "cuda":
        return 8
    if device == "mps":
        return 4
    return 1


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open this video: {video_path}\n"
            "Try converting it to MP4/H.264 first."
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Could not read video FPS. Convert to a constant-FPS MP4 first.")

    duration_s = total_frames / fps if total_frames > 0 else 0.0
    return VideoInfo(
        path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_s=duration_s,
    )


def create_run_dir(out_dir: Path, video_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("_") or "video"
    run_dir = out_dir / f"{safe_stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dataframe_from_detector_output(output: object) -> pd.DataFrame:
    if isinstance(output, pd.DataFrame):
        return output.copy()

    try:
        return pd.DataFrame(output)
    except Exception as exc:
        raise TypeError(f"Could not convert detector output to a DataFrame: {exc}") from exc


def _stub_torchcodec() -> None:
    """本机 torchcodec 导入会挂起；在 import feat 前注入 stub。"""
    import sys
    import types

    if "torchcodec.decoders" in sys.modules:
        return
    print("[C1] stub torchcodec (avoid hang)", flush=True)
    stub = types.ModuleType("torchcodec")
    decoders = types.ModuleType("torchcodec.decoders")

    class VideoDecoder:  # noqa: D401
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchcodec stubbed; use OpenCV frame path")

    decoders.VideoDecoder = VideoDecoder
    stub.decoders = decoders
    sys.modules["torchcodec"] = stub
    sys.modules["torchcodec.decoders"] = decoders


def _local_pyfeat_models_dir() -> Path:
    """仓库 data/pyfeat_models（curl 从 hf-mirror 预下载）。"""
    return Path(__file__).resolve().parents[6] / "data" / "pyfeat_models"


def _install_local_weight_hooks(models_dir: Path) -> Path | None:
    """用本地权重替换 huggingface_hub 下载，避免联网卡住。"""
    retina = models_dir / "retinaface_r34" / "model.safetensors"
    multitask = models_dir / "face_multitask_v2" / "face_multitask_v2.safetensors"
    if not retina.is_file() or not multitask.is_file():
        print(
            f"本地权重不完整: {models_dir}\n"
            "请先运行: bash RAL/experiment/C/C1/code/mirror/download_models_mirror.sh",
            flush=True,
        )
        return None

    import feat.utils as feat_utils

    _orig_hf = feat_utils.hf_hub_download_with_fallback

    def _patched(repo_id, filename, fallback_filename, cache_dir):
        if "retinaface" in repo_id:
            print(f"[local] retinaface ← {retina}", flush=True)
            return str(retina)
        return _orig_hf(repo_id, filename, fallback_filename, cache_dir)

    feat_utils.hf_hub_download_with_fallback = _patched
    print(f"[local] multitask ← {multitask}", flush=True)
    return multitask


def make_detector(device: str):
    _stub_torchcodec()
    models_dir = _local_pyfeat_models_dir()
    multitask_path = _install_local_weight_hooks(models_dir)

    try:
        from feat import Detectorv2
    except Exception as exc:
        raise RuntimeError(
            f"Could not import Detectorv2 from py-feat: {exc}"
        ) from exc

    kwargs = {"device": device}
    signature = inspect.signature(Detectorv2)
    if "identity_model" in signature.parameters:
        kwargs["identity_model"] = None
    if multitask_path is not None and "multitask_weights" in signature.parameters:
        kwargs["multitask_weights"] = str(multitask_path)

    print(f"[C1] Detectorv2 device={device} local_weights={multitask_path is not None}", flush=True)
    return Detectorv2(**kwargs)


def _extract_frames_opencv(video_path: Path, frame_step: int, tmp_dir: Path) -> list[Path]:
    """用 OpenCV 抽帧，绕过 torchcodec。"""
    ensure_runtime_dependencies()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    paths: list[Path] = []
    idx = 0
    saved = 0
    step = max(1, frame_step)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out = tmp_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            paths.append(out)
            saved += 1
            if saved % 30 == 0:
                print(f"[C1] extracted {saved} frames (idx={idx})", flush=True)
        idx += 1
    cap.release()
    print(f"[C1] extracted total {len(paths)} frames from {idx} (step={step})", flush=True)
    if not paths:
        raise RuntimeError("No frames extracted from video")
    return paths


def run_detector(
    video_path: Path,
    *,
    device: str,
    batch_size: int,
    frame_step: int,
    face_detection_threshold: float,
) -> pd.DataFrame:
    import tempfile
    import shutil

    detector = make_detector(device)
    tmp_dir = Path(tempfile.mkdtemp(prefix="c1_frames_"))
    try:
        print("[C1] extract frames via OpenCV...", flush=True)
        frame_paths = _extract_frames_opencv(video_path, frame_step, tmp_dir)
        print(f"[C1] detect {len(frame_paths)} images...", flush=True)
        kwargs = {
            "data_type": "image",
            "batch_size": batch_size,
            "face_detection_threshold": face_detection_threshold,
            "num_workers": 0,
            "progress_bar": True,
        }
        output = detector.detect([str(p) for p in frame_paths], **kwargs)
        df = dataframe_from_detector_output(output)
        if df.empty:
            raise RuntimeError("Py-Feat returned no rows. No face may have been detected.")

        # 从文件名恢复真实帧号：frame_000123.jpg
        if "input" in df.columns or "FileName" in df.columns:
            col = "input" if "input" in df.columns else "FileName"
            frames = []
            for name in df[col].astype(str):
                stem = Path(name).stem  # frame_000123
                try:
                    frames.append(int(stem.split("_")[-1]))
                except Exception:
                    frames.append(-1)
            df["frame"] = frames
        return df
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def select_main_face(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame_col = find_column(raw_df, ["frame"])
    score_col = find_column(raw_df, ["FaceScore", "face_score", "score"])

    work = raw_df.copy()
    work[frame_col] = pd.to_numeric(work[frame_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.sort_values([frame_col, score_col], ascending=[True, False])
    return work.drop_duplicates(subset=[frame_col], keep="first")


def build_frame_timeline(
    raw_df: pd.DataFrame,
    video_info: VideoInfo,
    *,
    face_threshold: float,
    smooth_ms: int,
    interpolate_limit_ms: int,
) -> pd.DataFrame:
    frame_col = find_column(raw_df, ["frame"])
    score_col = find_column(raw_df, ["FaceScore", "face_score", "score"])
    valence_col = find_column(raw_df, ["valence", "v"])
    arousal_col = find_column(raw_df, ["arousal", "a"])

    main_face = select_main_face(raw_df)
    compact = main_face[[frame_col, score_col, valence_col, arousal_col]].copy()
    compact.columns = ["frame", "face_score", "valence_raw", "arousal_raw"]

    for column in compact.columns:
        compact[column] = pd.to_numeric(compact[column], errors="coerce")

    compact["frame"] = compact["frame"].round().astype("Int64")

    if video_info.total_frames > 0:
        frames = np.arange(video_info.total_frames, dtype=int)
    else:
        max_frame = int(compact["frame"].dropna().max()) if compact["frame"].notna().any() else -1
        frames = np.arange(max_frame + 1, dtype=int)

    timeline = pd.DataFrame({"frame": frames})
    timeline = timeline.merge(compact, on="frame", how="left")
    timeline["timestamp_s"] = timeline["frame"] / video_info.fps
    timeline["timestamp_ms"] = timeline["timestamp_s"] * 1000.0
    timeline["fps"] = video_info.fps

    timeline["face_valid"] = (
        timeline["face_score"].notna()
        & (timeline["face_score"] >= face_threshold)
        & timeline["valence_raw"].notna()
        & timeline["arousal_raw"].notna()
    )
    timeline["valence"] = timeline["valence_raw"].where(timeline["face_valid"])
    timeline["arousal"] = timeline["arousal_raw"].where(timeline["face_valid"])

    limit_frames = max(0, int(round(video_info.fps * interpolate_limit_ms / 1000.0)))
    for column in ["valence", "arousal"]:
        filled = timeline[column].copy()
        if limit_frames > 0:
            filled = filled.interpolate(
                method="linear",
                limit=limit_frames,
                limit_area="inside",
            )
        timeline[f"{column}_filled"] = filled

    smooth_frames = max(1, int(round(video_info.fps * smooth_ms / 1000.0)))
    for column in ["valence_filled", "arousal_filled"]:
        timeline[column.replace("_filled", "_smooth")] = (
            timeline[column]
            .rolling(window=smooth_frames, min_periods=1, center=True)
            .mean()
        )

    ordered = [
        "frame",
        "timestamp_s",
        "timestamp_ms",
        "fps",
        "face_score",
        "face_valid",
        "valence_raw",
        "arousal_raw",
        "valence",
        "arousal",
        "valence_filled",
        "arousal_filled",
        "valence_smooth",
        "arousal_smooth",
    ]
    return timeline[ordered]


def save_excel(
    path: Path,
    sheets: dict[str, pd.DataFrame],
    metadata: dict[str, object] | None = None,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        if metadata:
            meta_df = pd.DataFrame(
                [{"key": key, "value": value} for key, value in metadata.items()]
            )
            meta_df.to_excel(writer, sheet_name="summary", index=False)


def plot_video_va(timeline: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot; matplotlib is not available: {exc}")
        return

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(timeline["timestamp_s"], timeline["valence_smooth"], label="valence smooth")
    ax.plot(timeline["timestamp_s"], timeline["arousal_smooth"], label="arousal smooth")
    ax.scatter(
        timeline.loc[timeline["face_valid"], "timestamp_s"],
        timeline.loc[timeline["face_valid"], "valence"],
        s=5,
        alpha=0.25,
        label="valid valence raw",
    )
    ax.set_title("Video VA Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VA")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def detect_source_columns(
    source_df: pd.DataFrame,
    *,
    time_col: str | None,
    valence_col: str | None,
    arousal_col: str | None,
) -> tuple[str, str, str, str]:
    if time_col is None:
        time_col = find_column(
            source_df,
            [
                "timestamp_s",
                "time_s",
                "seconds",
                "second",
                "time",
                "t",
                "timestamp_ms",
                "time_ms",
                "frame",
            ],
        )
    if valence_col is None:
        valence_col = find_column(
            source_df,
            ["source_valence", "original_valence", "target_valence", "valence", "v"],
        )
    if arousal_col is None:
        arousal_col = find_column(
            source_df,
            ["source_arousal", "original_arousal", "target_arousal", "arousal", "a"],
        )

    time_kind = "s"
    normalized_time = normalize_name(time_col)
    if "ms" in normalized_time:
        time_kind = "ms"
    elif normalized_time == "frame":
        time_kind = "frame"

    return time_col, valence_col, arousal_col, time_kind


def concordance_correlation_coefficient(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    x = x[valid].astype(float)
    y = y[valid].astype(float)
    if len(x) < 2:
        return float("nan")

    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var(ddof=0)
    var_y = y.var(ddof=0)
    cov = ((x - mean_x) * (y - mean_y)).mean()
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float((2 * cov) / denom) if denom else float("nan")


def compare_with_source(
    timeline: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    *,
    source_time_col: str | None,
    source_valence_col: str | None,
    source_arousal_col: str | None,
    source_fps: float | None,
    time_offset_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(source_csv)
    time_col, valence_col, arousal_col, time_kind = detect_source_columns(
        source,
        time_col=source_time_col,
        valence_col=source_valence_col,
        arousal_col=source_arousal_col,
    )

    work = source[[time_col, valence_col, arousal_col]].copy()
    work.columns = ["source_time", "source_valence", "source_arousal"]
    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["source_time"])

    if time_kind == "ms":
        work["source_timestamp_s"] = work["source_time"] / 1000.0
    elif time_kind == "frame":
        if source_fps is None:
            raise ValueError(
                "The source VA file uses frame numbers. Provide --source-fps to convert them to seconds."
            )
        work["source_timestamp_s"] = work["source_time"] / source_fps
    else:
        work["source_timestamp_s"] = work["source_time"]

    work["aligned_timestamp_s"] = work["source_timestamp_s"] + time_offset_ms / 1000.0

    detected = timeline.dropna(subset=["valence_smooth", "arousal_smooth"]).copy()
    if detected.empty:
        raise ValueError("No valid detected VA values are available for comparison.")

    detected_time = detected["timestamp_s"].to_numpy(dtype=float)
    detected_v = detected["valence_smooth"].to_numpy(dtype=float)
    detected_a = detected["arousal_smooth"].to_numpy(dtype=float)

    query_time = work["aligned_timestamp_s"].to_numpy(dtype=float)
    min_t = float(np.nanmin(detected_time))
    max_t = float(np.nanmax(detected_time))
    in_range = (query_time >= min_t) & (query_time <= max_t)

    work["detected_valence"] = np.nan
    work["detected_arousal"] = np.nan
    work.loc[in_range, "detected_valence"] = np.interp(
        query_time[in_range], detected_time, detected_v
    )
    work.loc[in_range, "detected_arousal"] = np.interp(
        query_time[in_range], detected_time, detected_a
    )

    work["valence_error"] = work["detected_valence"] - work["source_valence"]
    work["arousal_error"] = work["detected_arousal"] - work["source_arousal"]
    work["va_distance"] = np.sqrt(work["valence_error"] ** 2 + work["arousal_error"] ** 2)

    metrics_rows = []
    for name, detected_col, source_col, error_col in [
        ("valence", "detected_valence", "source_valence", "valence_error"),
        ("arousal", "detected_arousal", "source_arousal", "arousal_error"),
    ]:
        valid = work[[detected_col, source_col, error_col]].dropna()
        if valid.empty:
            metrics_rows.append(
                {
                    "metric": name,
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mean_error": np.nan,
                    "pearson_r": np.nan,
                    "ccc": np.nan,
                }
            )
            continue

        error = valid[error_col]
        metrics_rows.append(
            {
                "metric": name,
                "n": int(len(valid)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "mean_error": float(error.mean()),
                "pearson_r": float(valid[detected_col].corr(valid[source_col])),
                "ccc": concordance_correlation_coefficient(
                    valid[detected_col], valid[source_col]
                ),
            }
        )

    distance = work["va_distance"].dropna()
    metrics_rows.append(
        {
            "metric": "va_distance",
            "n": int(len(distance)),
            "mae": float(distance.mean()) if len(distance) else np.nan,
            "rmse": float(np.sqrt((distance**2).mean())) if len(distance) else np.nan,
            "mean_error": np.nan,
            "pearson_r": np.nan,
            "ccc": np.nan,
        }
    )

    metrics = pd.DataFrame(metrics_rows)
    work.to_csv(output_dir / "compare_aligned.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "compare_metrics.csv", index=False, encoding="utf-8-sig")
    plot_compare(work, output_dir / "compare_curve.png")
    return work, metrics


def plot_compare(aligned: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping comparison plot; matplotlib is not available: {exc}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=150, sharex=True)
    axes[0].plot(aligned["aligned_timestamp_s"], aligned["source_valence"], label="source")
    axes[0].plot(aligned["aligned_timestamp_s"], aligned["detected_valence"], label="detected")
    axes[0].set_ylabel("Valence")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(aligned["aligned_timestamp_s"], aligned["source_arousal"], label="source")
    axes[1].plot(aligned["aligned_timestamp_s"], aligned["detected_arousal"], label="detected")
    axes[1].set_ylabel("Arousal")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-frame Valence/Arousal values from a video with Py-Feat Detectorv2."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Root output folder. A timestamped subfolder will be created.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto chooses CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Default: 1 on CPU, 4 on MPS, 8 on CUDA.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame. 1 means every frame.",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=DEFAULT_FACE_THRESHOLD,
        help="Minimum FaceScore for a valid VA value.",
    )
    parser.add_argument(
        "--smooth-ms",
        type=int,
        default=DEFAULT_SMOOTH_MS,
        help="Centered rolling average window in milliseconds.",
    )
    parser.add_argument(
        "--interpolate-limit-ms",
        type=int,
        default=DEFAULT_INTERPOLATE_LIMIT_MS,
        help="Fill short missing gaps up to this many milliseconds before smoothing.",
    )
    parser.add_argument("--source-va", default=None, help="Optional source/original VA CSV.")
    parser.add_argument("--source-time-col", default=None, help="Time column in source VA CSV.")
    parser.add_argument(
        "--source-valence-col", default=None, help="Valence column in source VA CSV."
    )
    parser.add_argument(
        "--source-arousal-col", default=None, help="Arousal column in source VA CSV."
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="FPS for source VA CSV when its time column is frame.",
    )
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help="Add this offset to source timestamps before comparison.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not create XLSX output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_runtime_dependencies()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be 1 or greater.")

    out_root = Path(args.out_dir).expanduser().resolve()
    run_dir = create_run_dir(out_root, video_path)

    print("Input video:", video_path)
    print("Output folder:", run_dir)
    print("Reading video info...")
    video_info = read_video_info(video_path)
    print(
        f"Video: {video_info.width}x{video_info.height}, "
        f"{video_info.fps:.3f} fps, {video_info.total_frames} frames, "
        f"{video_info.duration_s:.2f}s"
    )

    device = choose_device(args.device)
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size(device)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Frame step:", args.frame_step)
    print("Initializing Py-Feat Detectorv2. First run may download model weights...")

    raw_df = run_detector(
        video_path,
        device=device,
        batch_size=batch_size,
        frame_step=args.frame_step,
        face_detection_threshold=min(args.face_threshold, 0.5),
    )
    raw_csv = run_dir / "video_va_full.csv"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print("Saved full detector output:", raw_csv)

    timeline = build_frame_timeline(
        raw_df,
        video_info,
        face_threshold=args.face_threshold,
        smooth_ms=args.smooth_ms,
        interpolate_limit_ms=args.interpolate_limit_ms,
    )
    timeline_csv = run_dir / "video_va.csv"
    timeline.to_csv(timeline_csv, index=False, encoding="utf-8-sig")
    print("Saved per-frame VA:", timeline_csv)

    plot_video_va(timeline, run_dir / "video_va_curve.png")

    detected_rows = int(timeline["face_score"].notna().sum())
    valid_rows = int(timeline["face_valid"].sum())
    total_rows = int(len(timeline))
    summary = RunSummary(
        video=video_info,
        device=device,
        batch_size=batch_size,
        frame_step=args.frame_step,
        face_threshold=args.face_threshold,
        smooth_ms=args.smooth_ms,
        interpolate_limit_ms=args.interpolate_limit_ms,
        total_output_rows=total_rows,
        detected_rows=detected_rows,
        valid_rows=valid_rows,
        valid_ratio=float(valid_rows / total_rows) if total_rows else 0.0,
        output_dir=str(run_dir),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    summary_dict = asdict(summary)
    summary_dict["video"] = asdict(video_info)
    (run_dir / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aligned = None
    metrics = None
    if args.source_va:
        source_csv = Path(args.source_va).expanduser().resolve()
        if not source_csv.exists():
            raise FileNotFoundError(f"Source VA CSV does not exist: {source_csv}")
        print("Comparing with source VA:", source_csv)
        aligned, metrics = compare_with_source(
            timeline,
            source_csv,
            run_dir,
            source_time_col=args.source_time_col,
            source_valence_col=args.source_valence_col,
            source_arousal_col=args.source_arousal_col,
            source_fps=args.source_fps,
            time_offset_ms=args.time_offset_ms,
        )
        print("Saved comparison outputs.")

    if not args.no_excel:
        sheets = {
            "video_va": timeline,
            "full_columns_preview": pd.DataFrame({"column": list(raw_df.columns)}),
        }
        if aligned is not None and metrics is not None:
            sheets["compare_aligned"] = aligned
            sheets["compare_metrics"] = metrics
        save_excel(
            run_dir / "video_va_report.xlsx",
            sheets,
            metadata={
                "video": str(video_path),
                "valid_ratio": f"{summary.valid_ratio:.2%}",
                "device": device,
                "batch_size": batch_size,
                "frame_step": args.frame_step,
            },
        )

    print("")
    print("Finished.")
    print(f"Valid face ratio: {summary.valid_ratio:.2%}")
    if summary.valid_ratio < 0.80:
        print(
            "Warning: valid face ratio is below 80%. Check lighting, face size, angle, "
            "or whether the robot face can be detected as a human-like face."
        )
    print("Main CSV:", timeline_csv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print("")
        print("ERROR:", exc)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
