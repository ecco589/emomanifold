#!/usr/bin/env python3
"""
B1 全量分析：200 组 contextual triplets → 句级 V + 关键词 token 级 V/A/D。

用法（仓库根目录）:
  python RAL/experiment/B/B1/code/run_b1_full.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parent
B1_DIR = CODE_DIR.parent
E1_CODE = B1_DIR.parent.parent / "A" / "A1" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(E1_CODE))

from project_root import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from b1_core import load_e1_projector, project_sentence_valence  # noqa: E402
from run_b1_context import find_keyword_valence  # noqa: E402

DATA_PATH = B1_DIR / "data" / "context_triplets_en.json"
RESULT_PATH = B1_DIR / "result" / "b1_full_metrics.json"
DIMS = ("valence", "arousal", "dominance")
LABELS = ("sarcastic", "positive", "negative")


def project_vad(model_loader, projector, text: str) -> dict:
    _, coords, tokens = project_sentence_valence(
        model_loader, projector, text, aggregation="last"
    )
    return {"tokens": tokens, "coords": coords}


def main():
    with DATA_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset["items"]

    from src.model_loader import CausalModelLoader  # noqa: E402

    print(f"Loading model … ({len(items)} triplets)")
    loader = CausalModelLoader("qwen", device="cpu")
    loader.load()
    projector, report = load_e1_projector(loader)

    sent_rows: List[dict] = []
    kw_rows: List[dict] = []

    for item in items:
        gid = item["id"]
        for label in LABELS:
            text = item[label]
            v_sent, coords, tokens = project_sentence_valence(
                loader, projector, text, aggregation="last"
            )
            sent_rows.append({
                "id": gid,
                "type": label,
                "sentence_v": round(v_sent, 4),
                "intent": item.get("intent", ""),
            })

        for kw in item.get("keywords", []):
            row = {"id": gid, "keyword": kw, "intent": item.get("intent", "")}
            for label in LABELS:
                res = project_vad(loader, projector, item[label])
                hit = find_keyword_valence(res["tokens"], res["coords"]["valence"], kw)
                for dim in DIMS:
                    if hit:
                        idx = hit[0]
                        row[f"{dim}_{label}"] = round(res["coords"][dim][idx], 4)
                    else:
                        row[f"{dim}_{label}"] = None
                if hit:
                    row[f"tok_{label}"] = hit[1]
                else:
                    row[f"tok_{label}"] = None
            kw_rows.append(row)

    # aggregates
    def _group_sent(label):
        return [r["sentence_v"] for r in sent_rows if r["type"] == label]

    vs, vp, vn = _group_sent("sarcastic"), _group_sent("positive"), _group_sent("negative")
    mean = lambda xs: sum(xs) / len(xs)

    # per-id S-N
    by_id = {}
    for r in sent_rows:
        by_id.setdefault(r["id"], {})[r["type"]] = r["sentence_v"]
    s_minus_n = [by_id[i]["sarcastic"] - by_id[i]["negative"] for i in by_id]
    s_minus_p = [by_id[i]["sarcastic"] - by_id[i]["positive"] for i in by_id]
    both_neg = sum(
        1 for i in by_id
        if by_id[i]["sarcastic"] < 0 and by_id[i]["negative"] < 0
    )
    close_sn = sum(1 for d in s_minus_n if abs(d) < 0.15)

    # keyword: shared S+P (valence)
    shared_sp = [
        r for r in kw_rows
        if r.get("valence_sarcastic") is not None and r.get("valence_positive") is not None
    ]
    delta_sp = [r["valence_sarcastic"] - r["valence_positive"] for r in shared_sp]

    # primary keyword only (first in list per item)
    primary_kws = {}
    for item in items:
        kws = item.get("keywords", [])
        if kws:
            primary_kws[item["id"]] = kws[0]
    primary_rows = [r for r in kw_rows if r["keyword"] == primary_kws.get(r["id"])]
    delta_sp_primary = [
        r["valence_sarcastic"] - r["valence_positive"]
        for r in primary_rows
        if r.get("valence_sarcastic") is not None and r.get("valence_positive") is not None
    ]

    summary = {
        "experiment": "B1_full",
        "n_triplets": len(items),
        "model": "qwen2.5-0.5B",
        "axes": "E1 en/pole_50",
        "sentence_level": {
            "mean_v_sarcastic": round(mean(vs), 4),
            "mean_v_positive": round(mean(vp), 4),
            "mean_v_negative": round(mean(vn), 4),
            "mean_delta_s_minus_n": round(mean(s_minus_n), 4),
            "mean_delta_s_minus_p": round(mean(s_minus_p), 4),
            "fraction_both_negative": round(both_neg / len(by_id), 4),
            "fraction_abs_delta_sn_lt_0.15": round(close_sn / len(by_id), 4),
        },
        "keyword_level_valence": {
            "n_keyword_rows": len(kw_rows),
            "n_shared_s_p": len(shared_sp),
            "mean_delta_v_s_minus_p_all_kw": round(mean(delta_sp), 4) if delta_sp else None,
            "mean_delta_v_s_minus_p_primary_kw": round(mean(delta_sp_primary), 4) if delta_sp_primary else None,
        },
        "scaler": report.get("scaler"),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "sentence": sent_rows, "keywords": kw_rows},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== B1 Full (n=200) Summary ===")
    sl = summary["sentence_level"]
    kl = summary["keyword_level_valence"]
    print(f"Sentence V  S={sl['mean_v_sarcastic']:.3f}  P={sl['mean_v_positive']:.3f}  N={sl['mean_v_negative']:.3f}")
    print(f"Mean Δ(S−N)={sl['mean_delta_s_minus_n']:.3f}  Δ(S−P)={sl['mean_delta_s_minus_p']:.3f}")
    print(f"Both S,N < 0: {sl['fraction_both_negative']*100:.1f}%  |Δ(S−N)|<0.15: {sl['fraction_abs_delta_sn_lt_0.15']*100:.1f}%")
    print(f"Keyword ΔV(S−P) primary: {kl['mean_delta_v_s_minus_p_primary_kw']:.3f}  (all kw rows: {kl['mean_delta_v_s_minus_p_all_kw']:.3f})")
    print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
