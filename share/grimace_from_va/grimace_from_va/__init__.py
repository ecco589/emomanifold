"""Render a Grimace face from valence and arousal.

Weight order is always:
joy, surprise, fear, sadness, disgust, anger.

``va_to_weights(v, a)`` maps one point and does not open a browser.
``render_png(v, a, path)`` writes one still.
``render_video(csv_path, mp4_path)`` renders a series on the experiment clock.
"""
from __future__ import annotations

from .fkcn import (
    FKCN_LABELS,
    GRIMACE_LABELS,
    ema_smooth,
    fkcn_weights,
    self_check,
    to_grimace,
    va_to_weights,
)
from .renderer import GrimaceRenderer, frames_to_mp4
from .trajectory import (
    DEFAULT_ALPHA,
    DEFAULT_FPS,
    DEFAULT_MAX_DELTA,
    DEFAULT_SETTLE_FRAMES,
    DEFAULT_TRANSITION_MS,
    GrimaceTrajectory,
    MappedRows,
    display_trajectory,
    load_va_csv,
    map_rows,
)

__all__ = [
    "FKCN_LABELS",
    "GRIMACE_LABELS",
    "DEFAULT_ALPHA",
    "DEFAULT_FPS",
    "DEFAULT_TRANSITION_MS",
    "DEFAULT_MAX_DELTA",
    "DEFAULT_SETTLE_FRAMES",
    "MappedRows",
    "GrimaceTrajectory",
    "GrimaceRenderer",
    "fkcn_weights",
    "ema_smooth",
    "to_grimace",
    "va_to_weights",
    "load_va_csv",
    "map_rows",
    "display_trajectory",
    "render_png",
    "render_video",
    "frames_to_mp4",
    "self_check",
]


def render_png(
    valence: float,
    arousal: float,
    out: str,
    *,
    width: int = 640,
    height: int = 640,
    amplify: bool = True,
    clip: bool = True,
) -> str:
    """Write one still. Uses ``va_to_weights`` (no EMA, no fade-in)."""
    weights = va_to_weights(valence, arousal, clip=clip)
    with GrimaceRenderer(width=width, height=height, amplify=amplify) as renderer:
        renderer.save(weights, out)
    return out


def render_video(
    csv_path: str,
    out: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    fps: float = DEFAULT_FPS,
    lead_s: float = 0.0,
    trail_s: float = 0.0,
    width: int = 640,
    height: int = 640,
    amplify: bool = True,
    prefer_raw: bool = True,
    weights_out: str | None = None,
) -> str:
    """Render a VA csv with the experiment display path.

    See ``display_trajectory`` for the frame clock. Pass ``lead_s=1`` and
    ``trail_s=1`` to match the C1 experiment videos.
    """
    valence, arousal, timestamps, _tokens = load_va_csv(csv_path, prefer_raw=prefer_raw)
    traj = display_trajectory(
        valence,
        arousal,
        timestamps,
        alpha=alpha,
        fps=fps,
        lead_s=lead_s,
        trail_s=trail_s,
    )
    if weights_out:
        traj.save_csv(weights_out)
    pngs: list[bytes] = []
    with GrimaceRenderer(width=width, height=height, amplify=amplify) as renderer:
        for weights in traj.weights:
            pngs.append(renderer.render(weights.tolist()))
    frames_to_mp4(pngs, out, fps)
    return out
