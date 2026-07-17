#!/usr/bin/env python3
"""C1 工控机录制：Qwen 流式 + 实机表情 + USB 相机录像。"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# EmotionManifold project root (RAL/experiment/C1/code -> ../../../../)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
C1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ROBOT_MOVE_MS = 100
ROBOT_SMOOTH_TAU_S = 0.25
POST_TOKEN_DELAY_S = 0.5
INTER_TRIAL_DELAY_S = 2.5

SYSTEM_PROMPT = (
    "You are a Qwen chat assistant. Reply in English by default.\n"
    "Be concise and natural—usually one or two sentences. "
    "Answer the user's question directly. "
    "Do not use emojis or markdown code blocks."
)


class CameraRecorder:
    """USB 相机预览 + 后台帧采集/写盘。"""

    def __init__(
        self,
        device: str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        preview: bool = True,
        crop_ratio: float = 1.0,
        out_width: int = 1920,
        out_height: int = 1080,
    ):
        self.device = device
        self.capture_width = width
        self.capture_height = height
        self.width = out_width
        self.height = out_height
        self.fps = fps
        self.preview = preview
        self.crop_ratio = max(0.1, min(1.0, crop_ratio))
        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest = None
        self._recording = False
        self._out_path: Optional[Path] = None

    def open(self) -> None:
        src = self.device
        if src.isdigit():
            self._cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机: {self.device}")
        # 1080p YUYV 仅 5fps；MJPG 可达 30fps（Mirror 要求）
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.capture_width, self.capture_height, self.fps = actual_w, actual_h, float(actual_fps)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.crop_ratio < 1.0:
            cw = max(2, int(w * self.crop_ratio))
            ch = max(2, int(h * self.crop_ratio))
            x0 = (w - cw) // 2
            y0 = (h - ch) // 2
            frame = frame[y0 : y0 + ch, x0 : x0 + cw]
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = self._process_frame(frame)
            with self._lock:
                self._latest = frame
                if self._recording and self._ffmpeg is not None and self._ffmpeg.stdin:
                    try:
                        self._ffmpeg.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        pass
            if self.preview:
                show = frame
                # Scale down for display only — never crop; show full camera FOV
                max_w = 1280
                if show.shape[1] > max_w:
                    scale = max_w / show.shape[1]
                    show = cv2.resize(
                        show,
                        (max_w, int(show.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("C1 Camera Preview (full view, q=quit)", show)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop.set()
                    break

    def start_recording(self, out_path: Path) -> float:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("需要 ffmpeg 才能输出 H.264 MP4（sudo apt install ffmpeg）")
        self._out_path = out_path
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self._ffmpeg = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        t0 = time.monotonic()
        with self._lock:
            self._recording = True
        return t0

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
        if self._ffmpeg is not None:
            if self._ffmpeg.stdin:
                self._ffmpeg.stdin.close()
            self._ffmpeg.wait(timeout=30)
            if self._ffmpeg.returncode != 0:
                raise RuntimeError(f"ffmpeg 编码失败 (code {self._ffmpeg.returncode})")
            self._ffmpeg = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.stop_recording()
        if self._cap is not None:
            self._cap.release()
        if self.preview:
            cv2.destroyAllWindows()


def load_stimuli(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def save_source_va(path: Path, records: List[Tuple[float, float, float, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "valence", "arousal", "token"])
        for ts, v, a, tok in records:
            w.writerow([f"{ts:.3f}", f"{v:.4f}", f"{a:.4f}", tok])


def run_trial(
    trial_idx: int,
    total: int,
    row: dict,
    out_dir: Path,
    model,
    tokenizer,
    projector,
    fkcn,
    device: str,
    robot_sink,
    robot_motion,
    camera: CameraRecorder,
) -> None:
    trial_id = row["id"]
    category = row.get("category", "")
    user_prompt = row["user_prompt"]
    trial_dir = out_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    video_path = trial_dir / "video.mp4"
    va_path = trial_dir / "source_va.csv"
    session_path = trial_dir / "session.json"

    print(f"[C1] Trial {trial_idx} / {total}  id={trial_id}  category={category}")
    print(f"     user: {user_prompt}")
    print("     status: recording | generating", flush=True)

    from src.stream_engine import generate_with_face_stream

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    robot_sink.set_enabled(True)
    robot_motion.snap_to_neutral()
    time.sleep(0.3)

    t0 = camera.start_recording(video_path)
    va_records: List[Tuple[float, float, float, str]] = []
    assistant_parts: List[str] = []

    for token_str, vad, _weights in generate_with_face_stream(
        model,
        tokenizer,
        projector,
        messages,
        fkcn,
        device=device,
        max_new_tokens=96,
        temperature=0.4,
    ):
        ts = time.monotonic() - t0
        vv = float(vad.get("valence", 0.0))
        aa = float(vad.get("arousal", 0.0))
        va_records.append((ts, vv, aa, token_str))
        assistant_parts.append(token_str)
        robot_motion.set_target(vv, aa)

    time.sleep(POST_TOKEN_DELAY_S)
    duration_s = time.monotonic() - t0
    camera.stop_recording()

    print("     status: saving", flush=True)
    save_source_va(va_path, va_records)

    session = {
        "trial_id": trial_id,
        "category": category,
        "user_prompt": user_prompt,
        "assistant_text": "".join(assistant_parts),
        "duration_s": round(duration_s, 3),
        "n_tokens": len(va_records),
        "video_fps": camera.fps,
        "camera_device": camera.device,
        "video_resolution": [camera.width, camera.height],
        "capture_resolution": [camera.capture_width, camera.capture_height],
        "crop_ratio": camera.crop_ratio,
    }
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"     saved: {video_path.name}, {va_path.name}, {session_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C1 实机 Mirror 录制")
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=C1_ROOT / "data" / "stimuli_one.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=C1_ROOT / "result" / "recordings")
    parser.add_argument("--camera", type=str, default="/dev/video0")
    parser.add_argument("--start-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--crop-ratio", type=float, default=1.0,
                        help="Center crop fraction of capture frame (1.0 = full view)")
    parser.add_argument("--out-width", type=int, default=1920)
    parser.add_argument("--out-height", type=int, default=1080)
    args = parser.parse_args()

    camera = CameraRecorder(
        args.camera,
        preview=not args.no_preview,
        crop_ratio=args.crop_ratio,
        out_width=args.out_width,
        out_height=args.out_height,
    )
    camera.open()
    print(
        f"相机就绪: {args.camera}  capture={camera.capture_width}x{camera.capture_height} "
        f"→ output={camera.width}x{camera.height} (crop={camera.crop_ratio:.0%}) "
        f"@ {camera.fps:.1f}fps"
    )

    if args.preview_only:
        print("预览模式（按 q 退出）…")
        try:
            while not camera._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        camera.close()
        return

    print("加载 Qwen + 实机…", flush=True)
    from src.stream_engine import load_chat_model_and_projector, load_fkcn
    from src.robot_face_sink import RobotFaceSink
    from src.robot_motion_smoother import RobotMotionSmoother

    model, tokenizer, projector = load_chat_model_and_projector("qwen")
    fkcn = load_fkcn(device="cpu")
    device = str(next(model.parameters()).device)
    robot_sink = RobotFaceSink()
    robot_motion = RobotMotionSmoother(
        robot_sink,
        fps=1000.0 / ROBOT_MOVE_MS,
        move_ms=ROBOT_MOVE_MS,
        smooth_tau_s=ROBOT_SMOOTH_TAU_S,
    )
    print(f"实机: {robot_sink.set_enabled(True)}", flush=True)

    stimuli = load_stimuli(args.stimuli)
    if args.start_id:
        start_num = int(args.start_id)
        stimuli = [s for s in stimuli if int(s["id"]) >= start_num]
    if args.limit:
        stimuli = stimuli[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(stimuli)

    try:
        for i, row in enumerate(stimuli, start=1):
            run_trial(
                i, total, row, args.out_dir,
                model, tokenizer, projector, fkcn, device,
                robot_sink, robot_motion, camera,
            )
            if i < total:
                robot_motion.snap_to_neutral()
                time.sleep(INTER_TRIAL_DELAY_S)
    finally:
        robot_sink.set_enabled(False)
        camera.close()

    print(f"\n[C1] 完成 {total} 条 → {args.out_dir}")


if __name__ == "__main__":
    main()
