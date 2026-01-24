"""
FKCN (Fuzzy Kohonen Clustering Network)
将情绪坐标 (alpha, beta) 映射到 7 个基本表情的融合权重
"""
from typing import Iterable, List, Sequence, Tuple
import torch


EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "fear",
    "sadness",
    "disgust",
    "anger",
]


def _one_hot(index: int, size: int) -> List[int]:
    vec = [0] * size
    vec[index] = 1
    return vec


DEFAULT_PROTOTYPES: List[Tuple[Tuple[float, float], List[int]]] = [
    ((0.61, -0.47), _one_hot(0, len(EMOTION_LABELS))),  # Neutral
    ((0.81, 0.66), _one_hot(1, len(EMOTION_LABELS))),   # Happiness
    ((0.88, 0.59), _one_hot(1, len(EMOTION_LABELS))),   # Happiness
    ((-0.75, 0.61), _one_hot(2, len(EMOTION_LABELS))),  # Surprise
    ((-0.71, 0.58), _one_hot(2, len(EMOTION_LABELS))),  # Surprise
    ((-0.83, 0.54), _one_hot(3, len(EMOTION_LABELS))),  # Fear
    ((-0.91, 0.52), _one_hot(3, len(EMOTION_LABELS))),  # Fear
    ((-0.47, -0.56), _one_hot(4, len(EMOTION_LABELS))), # Sadness
    ((-0.54, -0.59), _one_hot(4, len(EMOTION_LABELS))), # Sadness
    ((-0.95, -0.06), _one_hot(5, len(EMOTION_LABELS))), # Disgust
    ((-0.84, -0.19), _one_hot(5, len(EMOTION_LABELS))), # Disgust
    ((-0.95, 0.17), _one_hot(6, len(EMOTION_LABELS))),  # Anger
    ((-0.98, 0.10), _one_hot(6, len(EMOTION_LABELS))),  # Anger
]


class FKCN:
    """FKCN权重计算器"""

    def __init__(
        self,
        prototypes: Sequence[Tuple[Tuple[float, float], Sequence[float]]] = DEFAULT_PROTOTYPES,
        device: str = "cpu",
    ):
        self.device = device
        coords = [p[0] for p in prototypes]
        weights = [p[1] for p in prototypes]
        self.prototype_coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        self.prototype_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _squared_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        diff = inputs[:, None, :] - self.prototype_coords[None, :, :]
        return (diff ** 2).sum(dim=-1)

    def _membership(self, distances: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        zero_mask = distances <= eps
        any_zero = zero_mask.any(dim=1)

        membership = torch.zeros_like(distances)
        if any_zero.any():
            zero_counts = zero_mask.sum(dim=1).clamp(min=1).float()
            membership[zero_mask] = 1.0
            membership[any_zero] = membership[any_zero] / zero_counts[any_zero, None]

        non_zero_mask = ~any_zero
        if non_zero_mask.any():
            d = distances[non_zero_mask]
            inv_d = 1.0 / (d + eps)
            denom = d * inv_d.sum(dim=1, keepdim=True)
            membership[non_zero_mask] = 1.0 / denom

        return membership

    def compute_weights_batch(self, points: Iterable[Tuple[float, float]]) -> List[List[float]]:
        inputs = torch.tensor(list(points), dtype=torch.float32, device=self.device)
        if inputs.dim() != 2 or inputs.shape[1] != 2:
            raise ValueError("FKCN输入应为形状 [n, 2] 的 (alpha, beta) 序列")

        distances = self._squared_distances(inputs)
        membership = self._membership(distances)
        fusion_weights = membership @ self.prototype_weights
        return fusion_weights.cpu().tolist()
