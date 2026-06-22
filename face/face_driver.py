# -*- coding: utf-8 -*-
"""
FaceDriver: VA -> FKCN 7 weights -> fuse 7 expression presets (factor space) -> servos.

Self-contained for the industrial PC (Linux):
- No project dependency outside this folder.
- pyserial / Motor / Communication are imported lazily, only when actually sending,
  so this module runs on macOS (dry-run) without those installed.

Python 3.8+ compatible (no `|` unions, no match, typing.* annotations).
"""
import os
import sys
import platform
from typing import Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Configuration (edit here on the industrial PC)
# --------------------------------------------------------------------------- #
SERIAL_PORT = "/dev/usb_face"   # Linux serial port for the face board
BAUDRATE = 9600
SERVO_NUM = 29
MOVE_TIME_MS = 300              # servo move duration per update
AUTO_CHMOD = False             # set True on Linux if the port needs chmod 777

CONFIG_YAML = "29_servo_config.yaml"
EXPRESSION_DIR = "expression29"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Minimal FKCN (kept identical to src/fkcn.py prototypes)
# FKCN order: neutral, happiness, surprise, fear, sadness, disgust, anger
# --------------------------------------------------------------------------- #
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "fear", "sadness", "disgust", "anger",
]


def _one_hot(index, size=7):
    v = [0.0] * size
    v[index] = 1.0
    return v


_PROTOTYPES = [
    ((0.61, -0.47), _one_hot(0)),
    ((0.81, 0.66), _one_hot(1)),
    ((0.88, 0.59), _one_hot(1)),
    ((-0.75, 0.61), _one_hot(2)),
    ((-0.71, 0.58), _one_hot(2)),
    ((-0.83, 0.54), _one_hot(3)),
    ((-0.91, 0.52), _one_hot(3)),
    ((-0.47, -0.56), _one_hot(4)),
    ((-0.54, -0.59), _one_hot(4)),
    ((-0.95, -0.06), _one_hot(5)),
    ((-0.84, -0.19), _one_hot(5)),
    ((-0.95, 0.17), _one_hot(6)),
    ((-0.98, 0.10), _one_hot(6)),
]


def compute_fkcn(v, a):
    # type: (float, float) -> List[float]
    """Map (valence, arousal) to 7-dim basic-emotion fusion weights (sum = 1)."""
    eps = 1e-12
    dists = []
    for (cv, ca), _ in _PROTOTYPES:
        dv = v - cv
        da = a - ca
        dists.append(dv * dv + da * da)

    zero_idx = [i for i, d in enumerate(dists) if d <= eps]
    mem = [0.0] * len(_PROTOTYPES)
    if zero_idx:
        share = 1.0 / len(zero_idx)
        for i in zero_idx:
            mem[i] = share
    else:
        inv = [1.0 / (d + eps) for d in dists]
        inv_sum = sum(inv)
        for i, d in enumerate(dists):
            mem[i] = 1.0 / (d * inv_sum)

    weights = [0.0] * 7
    for i, (_, w_vec) in enumerate(_PROTOTYPES):
        m = mem[i]
        for j in range(7):
            weights[j] += m * w_vec[j]
    return weights


# Map FKCN emotion -> expression yaml filename (without extension)
EMOTION_TO_FILE = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "fear": "fear",
    "sadness": "sad",
    "disgust": "disgust",
    "anger": "angry",
}


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class FaceDriver:
    """Loads config + presets, fuses by weight, and sends (or dry-runs)."""

    def __init__(self, dry_run=None):
        # type: (Optional[bool]) -> None
        self.servo_num = SERVO_NUM
        self.start_deg = [0.0] * SERVO_NUM
        self.end_deg = [0.0] * SERVO_NUM
        self.index_list = list(range(SERVO_NUM))
        self.preset_factors = {}  # type: Dict[str, List[float]]
        self._controller = None

        self._load_config()
        self._load_presets()

        if dry_run is None:
            dry_run = platform.system() != "Linux"
        self.dry_run = dry_run

    # ------------------------------------------------------------------ #
    def _load_config(self):
        path = os.path.join(_CURRENT_DIR, CONFIG_YAML)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for name, info in cfg.items():
            idx = info["channel_idx"]
            self.start_deg[idx] = float(info["start_deg"])
            self.end_deg[idx] = float(info["end_deg"])

    def _angle_to_factor(self, angle, idx):
        diff = self.end_deg[idx] - self.start_deg[idx]
        if abs(diff) < 1e-9:
            return 0.0
        return _clamp01((angle - self.start_deg[idx]) / diff)

    def _load_presets(self):
        for emotion, fname in EMOTION_TO_FILE.items():
            path = os.path.join(_CURRENT_DIR, EXPRESSION_DIR, fname + ".yaml")
            with open(path, "r", encoding="utf-8") as f:
                angles = yaml.safe_load(f)
            if not isinstance(angles, list) or len(angles) < SERVO_NUM:
                raise ValueError("preset %s expects %d angles" % (path, SERVO_NUM))
            factors = [self._angle_to_factor(angles[i], i) for i in range(SERVO_NUM)]
            self.preset_factors[emotion] = factors

    # ------------------------------------------------------------------ #
    def fuse(self, weights7):
        # type: (List[float]) -> List[float]
        """Weighted blend of preset factors (weights need not be normalised)."""
        total = sum(weights7) or 1.0
        out = [0.0] * self.servo_num
        for k, emotion in enumerate(EMOTION_LABELS):
            w = weights7[k] / total
            if w == 0.0:
                continue
            preset = self.preset_factors[emotion]
            for i in range(self.servo_num):
                out[i] += w * preset[i]
        return [_clamp01(x) for x in out]

    def factors_from_va(self, v, a):
        # type: (float, float) -> Tuple[List[float], List[float]]
        weights7 = compute_fkcn(v, a)
        return weights7, self.fuse(weights7)

    # ------------------------------------------------------------------ #
    def _ensure_controller(self):
        if self._controller is not None:
            return self._controller
        if AUTO_CHMOD:
            os.system("sudo chmod 777 " + SERIAL_PORT)
        from Communication import UARTDevice
        from Motor import FaceController
        cfg_path = os.path.join(_CURRENT_DIR, CONFIG_YAML)
        self._controller = FaceController(
            UARTDevice(SERIAL_PORT, BAUDRATE), self.servo_num, cfg_path
        )
        return self._controller

    def apply_va(self, v, a, time_ms=MOVE_TIME_MS):
        # type: (float, float, int) -> List[float]
        weights7, factors = self.factors_from_va(v, a)
        self.apply_factors(factors, time_ms)
        return factors

    def apply_factors(self, factors, time_ms=MOVE_TIME_MS):
        # type: (List[float], int) -> None
        if self.dry_run:
            preview = ", ".join("%.2f" % x for x in factors)
            print("[dry-run] factors -> [%s]" % preview, flush=True)
            return
        controller = self._ensure_controller()
        controller.set_servo_angle_time_32(factors, self.index_list, time_ms)

    def reset(self, time_ms=MOVE_TIME_MS):
        self.apply_va(0.61, -0.47, time_ms)  # neutral prototype


# --------------------------------------------------------------------------- #
# Manual test (macOS dry-run): python face_driver.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    driver = FaceDriver()
    print("mode:", "dry-run" if driver.dry_run else "live", "| platform:", platform.system())
    samples = [
        ("neutral-ish", 0.61, -0.47),
        ("happy", 0.85, 0.62),
        ("sad", -0.50, -0.57),
        ("anger", -0.96, 0.13),
        ("origin", 0.0, 0.0),
    ]
    for name, v, a in samples:
        w7, factors = driver.factors_from_va(v, a)
        top = max(range(7), key=lambda i: w7[i])
        print("\n%s (V=%.2f, A=%.2f) -> top=%s" % (name, v, a, EMOTION_LABELS[top]))
        print("  w7   :", ["%.2f" % x for x in w7])
        print("  fac29:", ["%.2f" % x for x in factors])
