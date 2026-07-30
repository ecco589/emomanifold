"""
E3 核心：轴正交性、跨语言情绪簇 mean cosine、随机 3D 基线。
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

DIMS = ("valence", "arousal", "dominance")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def orthogonality(axes: Dict[str, torch.Tensor]) -> Dict[str, float]:
    def abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return abs(float(torch.dot(a.cpu(), b.cpu())))

    return {
        "va": round(abs_cos(axes["valence"], axes["arousal"]), 6),
        "ad": round(abs_cos(axes["arousal"], axes["dominance"]), 6),
        "vd": round(abs_cos(axes["valence"], axes["dominance"]), 6),
    }


def normalized_embedding(
    embedding_matrix: torch.Tensor,
    tokenizer,
    word: str,
) -> Optional[torch.Tensor]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        return None
    tid = ids[0]
    if tid >= embedding_matrix.shape[0]:
        return None
    vec = embedding_matrix[tid].float()
    return vec / (vec.norm() + 1e-8)


def project_emotion_3d(
    vec: torch.Tensor,
    axes: Dict[str, torch.Tensor],
    scaler_params: Dict[str, Dict[str, float]],
) -> torch.Tensor:
    coords = []
    for dim in DIMS:
        raw = float(vec.dot(axes[dim].cpu()))
        sp = scaler_params[dim]
        coords.append(raw * sp["scale"] + sp["bias"])
    return torch.tensor(coords, dtype=torch.float32)


def mean_pairwise_cosine(vectors: List[torch.Tensor]) -> Optional[float]:
    if len(vectors) < 2:
        return None
    total = 0.0
    count = 0
    for a, b in combinations(vectors, 2):
        na = a / (a.norm() + 1e-8)
        nb = b / (b.norm() + 1e-8)
        total += float(na.dot(nb))
        count += 1
    return total / count if count else None


def evaluate_clusters(
    embedding_matrix: torch.Tensor,
    tokenizer,
    clusters: List[dict],
    axes: Dict[str, torch.Tensor],
    scaler_params: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    per_cluster: Dict[str, Optional[float]] = {}
    skipped: List[str] = []
    for cluster in clusters:
        cid = cluster["id"]
        words = list(cluster["words"].values())
        vecs_3d = []
        for w in words:
            emb = normalized_embedding(embedding_matrix, tokenizer, w)
            if emb is None:
                skipped.append(f"{cid}:{w}")
                continue
            vecs_3d.append(project_emotion_3d(emb, axes, scaler_params))
        val = mean_pairwise_cosine(vecs_3d)
        per_cluster[cid] = round(val, 6) if val is not None else None

    values = [v for v in per_cluster.values() if v is not None]
    mean_val = round(sum(values) / len(values), 6) if values else None
    per_cluster["mean"] = mean_val
    return per_cluster, skipped


def random_baseline_clusters(
    embedding_matrix: torch.Tensor,
    tokenizer,
    clusters: List[dict],
    n_trials: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Per-cluster random 3D baseline; matches exploratory raxis()/proj() loop."""
    emb = embedding_matrix.float()
    emb_dim = emb.shape[1]
    if seed is not None:
        torch.manual_seed(seed)

    def raxis() -> Dict[str, torch.Tensor]:
        q, _ = torch.linalg.qr(torch.randn(emb_dim, 3))
        q = q.T
        return {DIMS[i]: q[i] / (q[i].norm() + 1e-8) for i in range(3)}

    def proj_word(word: str, axes: Dict[str, torch.Tensor]) -> torch.Tensor:
        ids = tokenizer.encode(word, add_special_tokens=False)
        vec = emb[ids[0]]
        vec = vec / (vec.norm() + 1e-8)
        return torch.tensor(
            [float(vec.dot(axes[d])) for d in DIMS],
            dtype=torch.float32,
        )

    per_cluster: Dict[str, float] = {}
    for cluster in clusters:
        cid = cluster["id"]
        words = list(cluster["words"].values())
        trial_vals: List[float] = []
        for _ in range(n_trials):
            axes = raxis()
            vecs = [proj_word(w, axes) for w in words]
            val = mean_pairwise_cosine(vecs)
            if val is not None:
                trial_vals.append(val)
        per_cluster[cid] = round(
            sum(trial_vals) / len(trial_vals), 6
        ) if trial_vals else 0.0

    per_cluster["mean"] = round(
        sum(per_cluster[c["id"]] for c in clusters) / len(clusters), 6
    )
    return per_cluster


def summarize_condition(
    name: str,
    pole_lexicon_path: str,
    scaler_params: Dict[str, Dict[str, float]],
    orth: Dict[str, float],
    cluster_cos: Dict[str, Optional[float]],
    skipped: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    payload = {
        "name": name,
        "pole_lexicon": pole_lexicon_path,
        "scaler": scaler_params,
        "orthogonality": orth,
        "cluster_cos": cluster_cos,
        "cluster_skipped": skipped,
    }
    if extra:
        payload.update(extra)
    return payload
