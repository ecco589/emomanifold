"""Valence–arousal → FKCN emotion weights.

Same prototypes and membership as ``src/fkcn.py`` (prototype set P=13).
NumPy only, so this package does not need PyTorch.
"""
from __future__ import annotations

import numpy as np

FKCN_LABELS: tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
)

# Grimace drops neutral. Index 1..6 line up with these names.
GRIMACE_LABELS: tuple[str, ...] = (
    "joy",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
)

# (valence, arousal) prototypes. Two anchors per emotion, plus neutral.
# Order matches FKCN_LABELS via the one-hot column.
_PROTO_VA = np.array(
    [
        [0.61, -0.47],  # neutral
        [0.81, 0.66],  # happiness
        [0.88, 0.59],  # happiness
        [-0.75, 0.61],  # surprise
        [-0.71, 0.58],  # surprise
        [-0.83, 0.54],  # fear
        [-0.91, 0.52],  # fear
        [-0.47, -0.56],  # sadness
        [-0.54, -0.59],  # sadness
        [-0.95, -0.06],  # disgust
        [-0.84, -0.19],  # disgust
        [-0.95, 0.17],  # anger
        [-0.98, 0.10],  # anger
    ],
    dtype=np.float64,
)
_PROTO_CLASS = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64)
_PROTO_W = np.eye(len(FKCN_LABELS), dtype=np.float64)[_PROTO_CLASS]

# Checked against src.fkcn.FKCN (torch, float32) on these points.
_REFERENCE = (
    ((0.81, 0.66), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0), (0.107593, 0.115283, 0.144177, 0.123153, 0.219107, 0.156437, 0.13425)),
    ((-0.5, -0.5), (0.002382, 0.001909, 0.004694, 0.004897, 0.95535, 0.021286, 0.009482)),
    ((0.2, 0.8), (0.047321, 0.381553, 0.185945, 0.138992, 0.070654, 0.081748, 0.093788)),
    ((-0.9, 0.2), (0.000898, 0.001519, 0.026422, 0.044225, 0.006462, 0.050675, 0.869799)),
    ((1.0, -1.0), (0.425119, 0.138337, 0.066513, 0.063071, 0.15066, 0.084826, 0.071475)),
)


def fkcn_weights(valence, arousal, *, clip: bool = True) -> np.ndarray:
    """Map VA points to 7-D FKCN weights.

    Parameters
    ----------
    valence, arousal:
        Scalars or same-length sequences. Range is nominally [-1, 1].
    clip:
        Clip each axis to [-1, 1] before the mapping.

    Returns
    -------
    ndarray, shape (7,) or (N, 7)
        Columns: neutral, happiness, surprise, fear, sadness, disgust, anger.
        Each row sums to 1.
    """
    v = np.asarray(valence, dtype=np.float64)
    a = np.asarray(arousal, dtype=np.float64)
    scalar = v.ndim == 0 and a.ndim == 0
    v = np.atleast_1d(v)
    a = np.atleast_1d(a)
    if v.shape != a.shape:
        raise ValueError(f"valence shape {v.shape} != arousal shape {a.shape}")
    if v.ndim != 1:
        raise ValueError("valence and arousal must be scalars or 1-D sequences")
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("valence and arousal must be finite")
    if clip:
        v = np.clip(v, -1.0, 1.0)
        a = np.clip(a, -1.0, 1.0)

    pts = np.stack([v, a], axis=1)
    diff = pts[:, None, :] - _PROTO_VA[None, :, :]
    dist = np.sum(diff * diff, axis=-1)

    eps = 1e-12
    membership = np.zeros_like(dist)
    zero = dist <= eps
    any_zero = zero.any(axis=1)
    if np.any(any_zero):
        counts = zero.sum(axis=1).astype(np.float64)
        membership[zero] = 1.0
        membership[any_zero] /= counts[any_zero, None]
    rest = ~any_zero
    if np.any(rest):
        d = dist[rest]
        inv = 1.0 / (d + eps)
        denom = d * inv.sum(axis=1, keepdims=True)
        membership[rest] = 1.0 / denom

    weights = membership @ _PROTO_W
    if scalar:
        return weights[0]
    return weights


def ema_smooth(weights, alpha: float = 0.6) -> np.ndarray:
    """EMA along the first axis. ``out[0] = weights[0]``.

    ``out[t] = alpha * weights[t] + (1 - alpha) * out[t - 1]``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("weights must have shape (N, D)")
    if len(w) == 0:
        return w.copy()
    out = np.empty_like(w)
    out[0] = w[0]
    keep = 1.0 - alpha
    for i in range(1, len(w)):
        out[i] = alpha * w[i] + keep * out[i - 1]
    return out


def to_grimace(fkcn, /) -> np.ndarray:
    """Drop the neutral column. 7-D FKCN → 6-D Grimace.

    Grimace order: joy, surprise, fear, sadness, disgust, anger.
    """
    w = np.asarray(fkcn, dtype=np.float64)
    if w.shape[-1] != 7:
        raise ValueError(f"expected 7 FKCN columns, got shape {w.shape}")
    return w[..., 1:]


def va_to_weights(valence: float, arousal: float, *, clip: bool = True) -> list[float]:
    """One VA point → six Grimace weights. No EMA, no fade-in.

    Order: joy, surprise, fear, sadness, disgust, anger.
    """
    return to_grimace(fkcn_weights(valence, arousal, clip=clip)).tolist()


def self_check(atol: float = 1e-4) -> None:
    """Raise AssertionError if the mapping drifts from the checked points."""
    for (v, a), expected in _REFERENCE:
        got = fkcn_weights(v, a)
        if not np.allclose(got, expected, atol=atol, rtol=0.0):
            raise AssertionError(f"FKCN({v}, {a}) = {got}, expected {expected}")
        if abs(float(got.sum()) - 1.0) > 1e-6:
            raise AssertionError(f"weights do not sum to 1: {got}")
    happy = va_to_weights(0.81, 0.66)
    if abs(happy[0] - 1.0) > 1e-6:
        raise AssertionError(f"happiness prototype should be pure joy, got {happy}")
