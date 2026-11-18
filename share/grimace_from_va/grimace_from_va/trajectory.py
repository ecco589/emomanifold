"""VA series → Grimace weights, including the 10 fps display path.

Display path (same constants as ``render_grimace_from_source_va.py``):

    clip → FKCN (P=13) → EMA (alpha=0.6)
      → 10 fps smoothstep, max step 0.05 → drop neutral → 6-D Grimace
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fkcn import FKCN_LABELS, GRIMACE_LABELS, ema_smooth, fkcn_weights, to_grimace

DEFAULT_ALPHA = 0.6
DEFAULT_FPS = 10.0
DEFAULT_TRANSITION_MS = 150.0
DEFAULT_MAX_DELTA = 0.05
DEFAULT_SETTLE_FRAMES = 10
# FaceFrameInterpolator default. The last token's frame budget is SETTLE_FRAMES;
# the interpolator itself stops settling after this many extra frames.
_MAX_SETTLE_FRAMES = 8

_TIME_COLUMNS = ("timestamp_s", "time_s", "time")
_TOKEN_COLUMNS = ("token", "text")


@dataclass
class MappedRows:
    """One Grimace weight vector per input VA row, after EMA."""

    valence: np.ndarray
    arousal: np.ndarray
    timestamps: np.ndarray | None
    tokens: list[str] | None
    fkcn: np.ndarray
    weights: np.ndarray

    def save_csv(self, path: str | Path) -> None:
        """Write one row per input sample (not per video frame)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["index", "timestamp_s", "valence", "arousal", "token"]
            header += [f"fkcn_{name}" for name in FKCN_LABELS]
            header += [f"grimace_{name}" for name in GRIMACE_LABELS]
            writer.writerow(header)
            n = len(self.valence)
            for i in range(n):
                ts = "" if self.timestamps is None else f"{self.timestamps[i]:.4f}"
                token = "" if not self.tokens else self.tokens[i]
                row = [
                    i,
                    ts,
                    f"{self.valence[i]:.6f}",
                    f"{self.arousal[i]:.6f}",
                    token,
                ]
                row += [f"{x:.6f}" for x in self.fkcn[i]]
                row += [f"{x:.6f}" for x in self.weights[i]]
                writer.writerow(row)


@dataclass
class GrimaceTrajectory:
    """Uniform-fps Grimace frames ready to render."""

    times: np.ndarray
    weights: np.ndarray
    fps: float
    token_fkcn: np.ndarray
    token_weights: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    timestamps: np.ndarray
    rate_limit_fraction: float

    def save_csv(self, path: str | Path) -> None:
        """Write one row per video frame."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *GRIMACE_LABELS])
            for t, w in zip(self.times, self.weights):
                writer.writerow([f"{t:.4f}", *[f"{x:.6f}" for x in w]])


def load_va_csv(path: str | Path, *, prefer_raw: bool = True) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, list[str] | None
]:
    """Read a VA csv.

    Required columns: ``valence`` and ``arousal``.
    If ``raw_valence`` / ``raw_arousal`` are present and ``prefer_raw`` is set,
    those are used instead. That matches the experiment files, where ``raw_*``
    is the projector output and ``valence`` / ``arousal`` may already be smoothed.

    Optional: ``timestamp_s`` (also accepts ``time_s`` or ``time``), ``token``.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        fields = set(reader.fieldnames)
        if prefer_raw and "raw_valence" in fields and "raw_arousal" in fields:
            v_key, a_key = "raw_valence", "raw_arousal"
        elif "valence" in fields and "arousal" in fields:
            v_key, a_key = "valence", "arousal"
        else:
            raise ValueError(
                f"{path} needs columns valence, arousal "
                "(or raw_valence, raw_arousal)"
            )
        t_key = next((k for k in _TIME_COLUMNS if k in fields), None)
        tok_key = next((k for k in _TOKEN_COLUMNS if k in fields), None)

        valence: list[float] = []
        arousal: list[float] = []
        times: list[float] = []
        tokens: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                valence.append(float(row[v_key]))
                arousal.append(float(row[a_key]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: bad VA value") from exc
            if t_key is not None:
                try:
                    times.append(float(row[t_key]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no}: bad {t_key}") from exc
            if tok_key is not None:
                tokens.append(row[tok_key] or "")

    if not valence:
        raise ValueError(f"{path} has no data rows")
    ts = np.asarray(times, dtype=np.float64) if t_key is not None else None
    tok = tokens if tok_key is not None else None
    return (
        np.asarray(valence, dtype=np.float64),
        np.asarray(arousal, dtype=np.float64),
        ts,
        tok,
    )


def map_rows(
    valence,
    arousal,
    timestamps=None,
    tokens: list[str] | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    clip: bool = True,
) -> MappedRows:
    """FKCN + EMA, one 6-D face per input row. No frame interpolation."""
    fkcn = fkcn_weights(valence, arousal, clip=clip)
    if fkcn.ndim == 1:
        fkcn = fkcn[None, :]
    smooth = ema_smooth(fkcn, alpha=alpha)
    v = np.clip(np.asarray(valence, dtype=np.float64), -1.0, 1.0) if clip else np.asarray(valence, dtype=np.float64)
    a = np.clip(np.asarray(arousal, dtype=np.float64), -1.0, 1.0) if clip else np.asarray(arousal, dtype=np.float64)
    v = np.atleast_1d(v)
    a = np.atleast_1d(a)
    ts = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
    if ts is not None and len(ts) != len(v):
        raise ValueError("timestamps length must match valence")
    if tokens is not None and len(tokens) != len(v):
        raise ValueError("tokens length must match valence")
    return MappedRows(
        valence=v,
        arousal=a,
        timestamps=ts,
        tokens=tokens,
        fkcn=smooth,
        weights=to_grimace(smooth),
    )


def _ease(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: list[float], b: list[float], t: float) -> list[float]:
    return [ai + (bi - ai) * t for ai, bi in zip(a, b)]


def _rate_limit(prev: list[float], nxt: list[float], max_delta: float) -> list[float]:
    out = []
    for p, n in zip(prev, nxt):
        d = n - p
        if d > max_delta:
            d = max_delta
        elif d < -max_delta:
            d = -max_delta
        out.append(p + d)
    return out


def _animate_to(state: list[float], target: list[float], n_trans: int, max_delta: float):
    """Yield frames the same way ``FaceFrameInterpolator.animate_to`` does.

    ``state`` is updated in place, including the one extra step the experiment
    loop produces when it breaks out of a generator it will not record.
    """
    tgt = [float(x) for x in target]
    if len(tgt) != len(state):
        state[:] = [0.0] * len(tgt)
    start = state[:]
    n = n_trans
    for step in range(1, n + 1):
        raw = _lerp(start, tgt, _ease(step / n))
        limited = _rate_limit(state, raw, max_delta)
        state[:] = limited
        yield state[:]
    for _ in range(_MAX_SETTLE_FRAMES):
        if all(abs(state[j] - tgt[j]) <= max_delta * 0.5 for j in range(len(tgt))):
            break
        limited = _rate_limit(state, tgt, max_delta)
        state[:] = limited
        yield state[:]


def _replay_display(
    token_times: np.ndarray,
    smooth_w: np.ndarray,
    *,
    fps: float,
    transition_ms: float,
    max_delta: float,
    settle_frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Port of ``replay_display_path`` in the experiment script.

    Interpolates the 7-D EMA weights. A new token cuts off the previous
    transition. Returns times, frames (N, 7), and the fraction of steps
    that hit the rate limit.
    """
    state = [0.0] * int(smooth_w.shape[1])
    frame_dt = 1.0 / fps
    n_trans = max(1, int(round(transition_ms / (1000.0 / fps))))
    frames: list[list[float]] = []
    times: list[float] = []
    clock = float(token_times[0])
    n = len(smooth_w)

    for i, target_row in enumerate(smooth_w):
        if i < n - 1:
            gap = float(token_times[i + 1] - token_times[i])
            budget = max(1, int(round(gap / frame_dt)))
        else:
            budget = settle_frames
        for j, frame in enumerate(
            _animate_to(state, [float(x) for x in target_row], n_trans, max_delta)
        ):
            if j >= budget:
                break
            frames.append(frame)
            times.append(clock)
            clock += frame_dt

    arr = np.asarray(frames, dtype=np.float64)
    if len(arr) < 2:
        saturation = 0.0
    else:
        steps = np.abs(np.diff(arr, axis=0))
        saturation = float(np.mean(steps >= max_delta - 1e-9))
    return np.asarray(times, dtype=np.float64), arr, saturation


def display_trajectory(
    valence,
    arousal,
    timestamps=None,
    tokens: list[str] | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    fps: float = DEFAULT_FPS,
    transition_ms: float = DEFAULT_TRANSITION_MS,
    max_delta: float = DEFAULT_MAX_DELTA,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    lead_s: float = 0.0,
    trail_s: float = 0.0,
    clip: bool = True,
) -> GrimaceTrajectory:
    """Build the video frame sequence used by the experiment renderer.

    Without timestamps, samples are spaced ``transition_ms`` apart so each
    token gets one full transition. The face eases in from all-zero weights,
    which is the display path, not ``va_to_weights``.

    ``lead_s`` / ``trail_s`` repeat the first / last face. The experiment
    videos use 1.0 and 1.0 so a downstream face detector has settle frames.
    This function defaults to 0.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    mapped = map_rows(
        valence, arousal, timestamps, tokens, alpha=alpha, clip=clip
    )
    if mapped.timestamps is None:
        token_times = np.arange(len(mapped.valence), dtype=np.float64) * (
            transition_ms / 1000.0
        )
    else:
        token_times = np.asarray(mapped.timestamps, dtype=np.float64)
        if np.any(np.diff(token_times) < 0):
            raise ValueError("timestamps must be non-decreasing")

    times, frames7, saturation = _replay_display(
        token_times,
        mapped.fkcn,
        fps=fps,
        transition_ms=transition_ms,
        max_delta=max_delta,
        settle_frames=settle_frames,
    )
    grimace = to_grimace(frames7)

    if lead_s > 0 and len(grimace):
        n_lead = max(1, int(round(lead_s * fps)))
        lead_t = times[0] - np.arange(n_lead, 0, -1) / fps
        times = np.concatenate([lead_t, times])
        grimace = np.concatenate([np.repeat(grimace[:1], n_lead, axis=0), grimace])
    if trail_s > 0 and len(grimace):
        n_trail = max(1, int(round(trail_s * fps)))
        trail_t = times[-1] + np.arange(1, n_trail + 1) / fps
        times = np.concatenate([times, trail_t])
        grimace = np.concatenate([grimace, np.repeat(grimace[-1:], n_trail, axis=0)])
    times = times - float(times[0])

    return GrimaceTrajectory(
        times=times,
        weights=grimace,
        fps=fps,
        token_fkcn=mapped.fkcn,
        token_weights=mapped.weights,
        valence=mapped.valence,
        arousal=mapped.arousal,
        timestamps=token_times,
        rate_limit_fraction=saturation,
    )
