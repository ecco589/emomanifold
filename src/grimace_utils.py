"""
Grimace 表情权重工具：FKCN→Grimace 转换、EMA 平滑
供 stream_engine、render_grimace_images 等复用
"""
from typing import List, Union, Optional


def fkcn_to_grimace_weights(fkcn_weights: Union[List[float], List[List[float]]]) -> Union[List[float], List[List[float]]]:
    """
    FKCN 7 维 → Grimace 6 维（去掉 neutral）
    FKCN 顺序: neutral, happiness, surprise, fear, sadness, disgust, anger
    Grimace 顺序: joy, surprise, fear, sadness, disgust, anger
    """
    if not fkcn_weights:
        return fkcn_weights

    if isinstance(fkcn_weights[0], (int, float)):
        w = fkcn_weights
        if len(w) < 7:
            raise ValueError("FKCN 权重维度不足")
        return [w[1], w[2], w[3], w[4], w[5], w[6]]

    result = []
    for w in fkcn_weights:
        if len(w) < 7:
            raise ValueError("FKCN 权重维度不足")
        result.append([w[1], w[2], w[3], w[4], w[5], w[6]])
    return result


def ema_blend(prev: List[float], cur: List[float], alpha: float = 0.6) -> List[float]:
    """EMA 平滑：blended = alpha * cur + (1 - alpha) * prev"""
    if not prev:
        return cur[:]
    return [alpha * c + (1 - alpha) * p for p, c in zip(prev, cur)]


def smooth_weights_batch(weights: List[List[float]], alpha: float = 0.6) -> List[List[float]]:
    """整句 batch 的 EMA 平滑，供 render_grimace_images 等复用"""
    if not weights:
        return []
    smoothed = [weights[0][:]]
    for idx in range(1, len(weights)):
        blended = ema_blend(smoothed[-1], weights[idx], alpha)
        smoothed.append(blended)
    return smoothed


class StreamingEMASmoother:
    """流式 EMA 平滑器，维护上一步权重状态"""

    def __init__(self, alpha: float = 0.6, initial: Optional[List[float]] = None):
        self.alpha = alpha
        self.prev: List[float] = list(initial) if initial else [0.0] * 6

    def update(self, cur: List[float]) -> List[float]:
        blended = ema_blend(self.prev, cur, self.alpha)
        self.prev = blended
        return blended
