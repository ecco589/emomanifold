"""
逐 token 表情帧间插值：固定帧率 + ease-in-out + 单帧变化上限。
"""
import time
from typing import Generator, Iterable, List, Optional


def _ease_in_out(t: float) -> float:
    """Smoothstep: 0→1 两端缓入缓出"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    return [ai + (bi - ai) * t for ai, bi in zip(a, b)]


def _rate_limit(prev: List[float], nxt: List[float], max_delta: float) -> List[float]:
    out = []
    for p, n in zip(prev, nxt):
        d = max(-max_delta, min(max_delta, n - p))
        out.append(p + d)
    return out


class FaceFrameInterpolator:
    """
    将离散 token 权重转为固定帧率连续轨迹。
    默认：10fps（与实机 MOVE_TIME_MS=100 对齐）、150ms ease-in-out 过渡。
    """

    def __init__(
        self,
        fps: float = 10.0,
        transition_ms: float = 150.0,
        max_delta: float = 0.05,
        n_dims: int = 6,
        realtime: bool = True,
        max_settle_frames: int = 8,
    ):
        self.fps = fps
        self.transition_ms = transition_ms
        self.max_delta = max_delta
        self.realtime = realtime
        self.max_settle_frames = max_settle_frames
        self._current: List[float] = [0.0] * n_dims

    def _frame_count(self) -> int:
        frame_dt_ms = 1000.0 / self.fps
        return max(1, int(round(self.transition_ms / frame_dt_ms)))

    def animate_to(self, target: Iterable[float]) -> Generator[List[float], None, None]:
        """从当前状态插值到 target，按 fps 节拍 yield 中间帧。"""
        tgt = [float(x) for x in target]
        if len(tgt) != len(self._current):
            self._current = [0.0] * len(tgt)

        n = self._frame_count()
        frame_dt = 1.0 / self.fps
        start = self._current[:]

        for i in range(1, n + 1):
            t = _ease_in_out(i / n)
            raw = _lerp(start, tgt, t)
            limited = _rate_limit(self._current, raw, self.max_delta)
            self._current = limited
            yield self._current[:]
            if self.realtime and i < n:
                time.sleep(frame_dt)

        # 收尾：若 rate limit 未完全到达，用额外帧缓慢贴近 target
        for _ in range(self.max_settle_frames):
            if all(abs(self._current[j] - tgt[j]) <= self.max_delta * 0.5 for j in range(len(tgt))):
                break
            limited = _rate_limit(self._current, tgt, self.max_delta)
            self._current = limited
            yield self._current[:]
            if self.realtime:
                time.sleep(frame_dt)
