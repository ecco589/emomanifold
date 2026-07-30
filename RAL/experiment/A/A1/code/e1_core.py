"""
E1 核心：轴提取、AdaptiveScaler（bipolar 词库）、Spearman 评测。
对齐 NIPS B1：embedding 投影 + scaler + hold-out Spearman。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from scipy.stats import spearmanr

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from project_root import find_project_root

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.emotion_projector import EmotionProjector  # noqa: E402


def extract_axes_from_embedding(
    embedding_matrix: torch.Tensor,
    tokenizer,
    lexicon_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    with open(lexicon_path, encoding="utf-8") as f:
        lex = json.load(f)
    groups = lex["word_groups"]
    emb = embedding_matrix.to(device).float()
    axes = {}
    pairs = [
        ("valence", "valence_positive", "valence_negative"),
        ("arousal", "arousal_high", "arousal_low"),
        ("dominance", "dominance_high", "dominance_low"),
    ]
    for dim, hi_key, lo_key in pairs:
        hi, lo = groups.get(hi_key, []), groups.get(lo_key, [])
        if not hi or not lo:
            continue
        hi_mean = _cluster_mean(emb, tokenizer, hi)
        lo_mean = _cluster_mean(emb, tokenizer, lo)
        axis = hi_mean - lo_mean
        axes[dim] = axis / (axis.norm() + 1e-8)
    return axes


def _cluster_mean(emb, tokenizer, words: List[str]) -> torch.Tensor:
    vecs = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = ids[0]
        if tid < emb.shape[0]:
            vecs.append(emb[tid])
    if not vecs:
        raise ValueError(f"No valid single-token embeddings for: {words[:5]}...")
    return torch.stack(vecs).mean(dim=0)


class E1AdaptiveScaler:
    """E1 用 AdaptiveScaler（支持 bipolar 词库 + enforce_nonnegative）。"""

    def __init__(self, vad_lexicon: dict, device: str = "cpu"):
        from src.adaptive_scaler import AdaptiveScaler

        self._inner = AdaptiveScaler(vad_lexicon, device=device)
        self.dimensions = self._inner.dimensions
        self.scales = self._inner.scales
        self.biases = self._inner.biases
        self.stats = self._inner.stats
        self.is_trained = False

    def train(self, corpus, model_loader, projector, verbose=False):
        self._inner.train(corpus, model_loader, projector, verbose=verbose)
        self.scales = self._inner.scales
        self.biases = self._inner.biases
        self.stats = self._inner.stats
        self.is_trained = True

    def apply(self, raw_value: float, dimension: str) -> float:
        return self._inner.apply(raw_value, dimension)

    def enforce_nonnegative_scales(self) -> Dict[str, Any]:
        result = self._inner.enforce_nonnegative_scales()
        self.scales = self._inner.scales
        self.biases = self._inner.biases
        return result


def finite_rho_p(rho, p) -> Tuple[Optional[float], Optional[float]]:
    def _one(x):
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    return _one(rho), _one(p)


def compute_spearman(
    embedding_matrix: torch.Tensor,
    axes: Dict[str, torch.Tensor],
    scaler_params: Dict[str, Dict[str, float]],
    holdout: dict,
    tokenizer,
) -> Tuple[dict, dict]:
    """
    NIPS B1：embedding 归一化后与轴点积，再仿射缩放，对 hold-out 算 Spearman。
    """
    emb = embedding_matrix.float().cpu()
    axes_cpu = {d: v.cpu() for d, v in axes.items()}
    per_axis = holdout["per_axis"]
    dims = ["valence", "arousal", "dominance"]

    nrc_by_dim = {d: [] for d in dims}
    our_by_dim = {d: [] for d in dims}
    used_words = {d: [] for d in dims}
    skipped = []

    for dim in dims:
        for word, wvad in per_axis[dim].items():
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                skipped.append(f"{dim}:{word}")
                continue
            tid = ids[0]
            if tid >= emb.shape[0]:
                skipped.append(f"{dim}:{word}:oob")
                continue
            vec = emb[tid]
            vec_norm = vec / (vec.norm() + 1e-8)
            raw = vec_norm.dot(axes_cpu[dim]).item()
            sp = scaler_params[dim]
            scaled = raw * sp["scale"] + sp["bias"]
            our_by_dim[dim].append(scaled)
            nrc_by_dim[dim].append(wvad[dim])
            used_words[dim].append(word)

    results = {}
    for dim in dims:
        rho, pval = spearmanr(nrc_by_dim[dim], our_by_dim[dim])
        rho_f, p_f = finite_rho_p(rho, pval)
        results[dim] = {
            "rho": round(rho_f, 6) if rho_f is not None else None,
            "p": float(p_f) if p_f is not None else None,
            "n": len(nrc_by_dim[dim]),
        }

    meta = {"used_words": used_words, "skipped": skipped}
    return results, meta


def run_single_pole_n(
    model_loader,
    nrc: dict,
    priority: dict,
    holdout: dict,
    pole_n: int,
    language: str,
    out_dir: Path,
    exclude: Set[str],
    allow_cross_pole_reuse: bool = False,
) -> dict:
    """一次 pole_n × language 的完整 B1 流程。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = model_loader.tokenizer

    # NIPS B1：极词与 Spearman hold-out 可重叠，不做互斥排除。
    exclude_all = exclude or set()

    from b1_lexicon import select_word_groups, build_lexicon_payload, build_corpus_lines

    word_groups = select_word_groups(
        pole_n,
        tokenizer,
        nrc,
        priority,
        exclude=exclude_all,
        allow_cross_pole_reuse=allow_cross_pole_reuse,
    )
    lex_payload = build_lexicon_payload(word_groups, nrc, language)
    lex_path = out_dir / "pole_lexicon.json"
    with lex_path.open("w", encoding="utf-8") as f:
        json.dump(lex_payload, f, indent=2, ensure_ascii=False)

    corpus = build_corpus_lines(word_groups, language)
    (out_dir / "training_corpus.txt").write_text("\n".join(corpus) + "\n", encoding="utf-8")

    emb = model_loader.get_output_embedding_matrix()
    axes = extract_axes_from_embedding(emb, tokenizer, str(lex_path), device="cpu")

    projector_raw = EmotionProjector(axes, scaler=None, device="cpu")
    scaler = E1AdaptiveScaler(lex_payload, device="cpu")
    scaler.train(corpus, model_loader, projector_raw, verbose=False)
    scaler_nonneg = scaler.enforce_nonnegative_scales()

    scaler_params = {
        d: {"scale": float(scaler.scales[d]), "bias": float(scaler.biases[d])}
        for d in axes
    }

    spearman, spearman_meta = compute_spearman(
        emb, axes, scaler_params, holdout, tokenizer
    )

    report = {
        "experiment": "E1",
        "language": language,
        "model": "qwen",
        "pole_n": pole_n,
        "holdout_source": holdout.get("description", ""),
        "pole_holdout_disjoint": False,
        "allow_cross_pole_reuse": allow_cross_pole_reuse,
        "word_groups": word_groups,
        "scaler": scaler_params,
        "scaler_nonneg_adjustment": scaler_nonneg,
        "spearman": spearman,
        "spearman_meta": spearman_meta,
        "holdout_n_per_axis": holdout.get("n_per_axis"),
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
