"""
B1 核心：iSarcasmEval SubTask C 讽刺句 vs 改写句的句级效价投影。

仅前向传播：Qwen2.5-0.5B hidden state → E1 en/pole_50 轴 + scaler → 句级 valence。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

E1_POLE_DIR = Path(__file__).resolve().parents[3] / "A" / "A1" / "result" / "en" / "pole_50"

_WORD_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class SarcasmPair:
    sarcastic: str
    rephrase: str
    surface_valence_sarcastic: float
    surface_valence_rephrase: float


def load_isarcasm_task_c(csv_path: Path) -> List[SarcasmPair]:
    """解析 SubTask C CSV，返回 (讽刺句, 非讽刺改写) 对。"""
    pairs: List[SarcasmPair] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t0 = row["text_0"].strip()
            t1 = row["text_1"].strip()
            sid = int(row["sarcastic_id"])
            if sid == 0:
                sarcastic, rephrase = t0, t1
            else:
                sarcastic, rephrase = t1, t0
            pairs.append(
                SarcasmPair(
                    sarcastic=sarcastic,
                    rephrase=rephrase,
                    surface_valence_sarcastic=0.0,
                    surface_valence_rephrase=0.0,
                )
            )
    return pairs


def _surface_valence(text: str, nrc: Dict[str, Dict[str, float]]) -> float:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    vals = [nrc[w]["valence"] for w in words if w in nrc]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def annotate_surface_valence(
    pairs: List[SarcasmPair],
    nrc: Dict[str, Dict[str, float]],
) -> List[SarcasmPair]:
    out: List[SarcasmPair] = []
    for p in pairs:
        out.append(
            SarcasmPair(
                sarcastic=p.sarcastic,
                rephrase=p.rephrase,
                surface_valence_sarcastic=_surface_valence(p.sarcastic, nrc),
                surface_valence_rephrase=_surface_valence(p.rephrase, nrc),
            )
        )
    return out


def select_positive_surface_pairs(
    pairs: List[SarcasmPair],
    n: int,
) -> List[SarcasmPair]:
    """
    选取讽刺句 NRC 表面效价高于改写句的对（字面积极、意图消极陷阱）。
    按表面效价差降序取前 n 对。
    """
    candidates = [
        p for p in pairs
        if p.surface_valence_sarcastic > p.surface_valence_rephrase
    ]
    candidates.sort(
        key=lambda p: p.surface_valence_sarcastic - p.surface_valence_rephrase,
        reverse=True,
    )
    if len(candidates) < n:
        raise RuntimeError(
            f"Only {len(candidates)} positive-surface sarcasm pairs found, need {n}"
        )
    return candidates[:n]


def load_e1_projector(model_loader, pole_dir: Path = E1_POLE_DIR):
    """从 E1 en/pole_50 加载轴向量与 scaler 参数，构建 EmotionProjector。"""
    import json
    import sys

    e1_code = pole_dir.parents[2] / "code"
    if str(e1_code) not in sys.path:
        sys.path.insert(0, str(e1_code))

    from e1_core import extract_axes_from_embedding  # noqa: E402
    from src.emotion_projector import EmotionProjector  # noqa: E402
    from src.adaptive_scaler import AdaptiveScaler  # noqa: E402

    lex_path = pole_dir / "pole_lexicon.json"
    report_path = pole_dir / "report.json"
    with lex_path.open(encoding="utf-8") as f:
        lex_payload = json.load(f)
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    emb = model_loader.get_output_embedding_matrix()
    axes = extract_axes_from_embedding(
        emb, model_loader.tokenizer, str(lex_path), device="cpu"
    )

    scaler = AdaptiveScaler(lex_payload, device="cpu")
    for dim, params in report["scaler"].items():
        scaler.scales[dim] = float(params["scale"])
        scaler.biases[dim] = float(params["bias"])
    scaler.is_trained = True

    projector = EmotionProjector(axes, scaler=scaler, device="cpu")
    return projector, report


def project_sentence_valence(
    model_loader,
    projector,
    text: str,
    aggregation: str = "last",
) -> Tuple[float, Dict[str, List[float]], List[str]]:
    """
    纯前向：句级标定效价。

    aggregation: "last" 末 token；"mean" 序列均值（与 outline 句级聚合一致）。
    """
    emotion_coords, tokens = projector.project_text(
        model_loader, text, apply_scaling=True
    )
    vals = emotion_coords["valence"]
    if not vals:
        return 0.0, emotion_coords, tokens
    if aggregation == "last":
        score = vals[-1]
    elif aggregation == "mean":
        score = sum(vals) / len(vals)
    else:
        raise ValueError(aggregation)
    return float(score), emotion_coords, tokens


def run_pair_batch(
    model_loader,
    projector,
    pairs: List[SarcasmPair],
    aggregation: str = "last",
) -> List[dict]:
    rows: List[dict] = []
    for i, pair in enumerate(pairs, 1):
        v_sarc, _, _ = project_sentence_valence(
            model_loader, projector, pair.sarcastic, aggregation
        )
        v_reph, _, _ = project_sentence_valence(
            model_loader, projector, pair.rephrase, aggregation
        )
        rows.append({
            "idx": i,
            "sarcastic": pair.sarcastic,
            "rephrase": pair.rephrase,
            "valence_sarcastic": round(v_sarc, 4),
            "valence_rephrase": round(v_reph, 4),
            "delta": round(v_sarc - v_reph, 4),
            "surface_valence_sarcastic": round(pair.surface_valence_sarcastic, 4),
            "surface_valence_rephrase": round(pair.surface_valence_rephrase, 4),
        })
    return rows
