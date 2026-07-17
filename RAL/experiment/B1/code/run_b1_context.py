#!/usr/bin/env python3
"""
B1 上下文补全句测试：10 组 × (讽刺 / 正面 / 反面)，对比句级与关键词 token 级 V 值。

用法（仓库根目录）:
  python RAL/experiment/B1/code/run_b1_context.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parent
B1_DIR = CODE_DIR.parent
E1_CODE = B1_DIR.parent / "E1" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(E1_CODE))

from project_root import find_project_root  # noqa: E402

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from b1_core import load_e1_projector, project_sentence_valence  # noqa: E402

# 10 组低门槛英文三元句；keywords 为跨语境对照的核心情绪词
TRIPLETS: List[dict] = [
    {
        "id": 1,
        "topic": "weather",
        "keywords": ["lovely", "awful"],
        "sarcastic": (
            "I stepped outside into cold rain and said, "
            '"What lovely weather we\'re having today!"'
        ),
        "positive": (
            "I stepped outside into warm sunshine and said, "
            '"What lovely weather we\'re having today!"'
        ),
        "negative": (
            "I stepped outside into cold rain and said, "
            '"This weather is awful — cold, wet, and miserable."'
        ),
    },
    {
        "id": 2,
        "topic": "prices",
        "keywords": ["love", "happy", "hate"],
        "sarcastic": (
            "At the shop, I looked at the price tag and said, "
            '"I just love paying this much for something so small."'
        ),
        "positive": (
            "At the shop, I looked at a gift I really wanted and said, "
            '"I\'m happy to pay this — it\'s worth it."'
        ),
        "negative": (
            "At the shop, I looked at the price tag and said, "
            '"I hate paying this much for something so small."'
        ),
    },
    {
        "id": 3,
        "topic": "early alarm",
        "keywords": ["Great", "hate"],
        "sarcastic": (
            "My alarm went off at 6 a.m. and I thought, "
            '"Great — this is exactly how I wanted to start the day."'
        ),
        "positive": (
            "My alarm went off at 6 a.m. for a trip I was excited about, and I thought, "
            '"Great — this is exactly how I wanted to start the day."'
        ),
        "negative": (
            "My alarm went off at 6 a.m. and I thought, "
            '"I hate starting the day this early."'
        ),
    },
    {
        "id": 4,
        "topic": "late friend",
        "keywords": ["perfect", "frustrating"],
        "sarcastic": (
            "I had been waiting for twenty minutes when my friend arrived and said, "
            '"Sorry I\'m late — perfect timing, as always."'
        ),
        "positive": (
            "My friend arrived right on time and said, "
            '"Sorry I\'m only just on time — I tried not to keep you waiting."'
        ),
        "negative": (
            "I had been waiting for twenty minutes when my friend arrived, and I said, "
            '"You\'re really late again — this is frustrating."'
        ),
    },
    {
        "id": 5,
        "topic": "messy room",
        "keywords": ["amazing", "mess"],
        "sarcastic": (
            "I opened the door to clothes and dishes everywhere and said, "
            '"Wow, this room looks amazing."'
        ),
        "positive": (
            "I opened the door after cleaning all morning and said, "
            '"Wow, this room looks amazing."'
        ),
        "negative": (
            "I opened the door to clothes and dishes everywhere and said, "
            '"This room is a mess."'
        ),
    },
    {
        "id": 6,
        "topic": "slow internet",
        "keywords": ["love", "painfully"],
        "sarcastic": (
            "The video kept stopping, so I said, "
            '"I really love how fast the internet is today."'
        ),
        "positive": (
            "The video loaded instantly, and I said, "
            '"I really love how fast the internet is today."'
        ),
        "negative": (
            "The video kept stopping, so I said, "
            '"The internet is painfully slow today."'
        ),
    },
    {
        "id": 7,
        "topic": "heavy workload",
        "keywords": ["relaxing", "stressful", "exhausting"],
        "sarcastic": (
            "I looked at my long to-do list and said, "
            '"What a relaxing day this is going to be."'
        ),
        "positive": (
            "I finished my tasks early and said, "
            '"What a relaxing day this is going to be."'
        ),
        "negative": (
            "I looked at my long to-do list and said, "
            '"This is going to be a stressful and exhausting day."'
        ),
    },
    {
        "id": 8,
        "topic": "bad food",
        "keywords": ["wonderful", "terrible"],
        "sarcastic": (
            "I took one bite of the cold, salty soup and said, "
            '"Mmm, this tastes wonderful."'
        ),
        "positive": (
            "I took a bite of the hot homemade soup and said, "
            '"Mmm, this tastes wonderful."'
        ),
        "negative": (
            "I took one bite of the cold, salty soup and said, "
            '"This tastes terrible."'
        ),
    },
    {
        "id": 9,
        "topic": "long queue",
        "keywords": ["fun", "hate"],
        "sarcastic": (
            "I stood in a long queue for thirty minutes and said, "
            '"Standing here is so much fun."'
        ),
        "positive": (
            "I stood in line with friends, chatting and laughing, and said, "
            '"Standing here is so much fun."'
        ),
        "negative": (
            "I stood in a long queue for thirty minutes and said, "
            '"I hate waiting in lines like this."'
        ),
    },
    {
        "id": 10,
        "topic": "broken plans",
        "keywords": ["Perfect", "ruined", "disappointment"],
        "sarcastic": (
            "It started raining right before our picnic, and I said, "
            '"Perfect — this is exactly what we needed."'
        ),
        "positive": (
            "The sun came out just before our picnic, and I said, "
            '"Perfect — this is exactly what we needed."'
        ),
        "negative": (
            "It started raining right before our picnic, and I said, "
            '"This ruined our plans — what a disappointment."'
        ),
    },
]

_TOKEN_NORM = re.compile(r"[^a-zA-Z']+")


def _norm_token(tok: str) -> str:
    return _TOKEN_NORM.sub("", tok).lower()


def find_keyword_valence(
    tokens: List[str],
    valences: List[float],
    keyword: str,
) -> Optional[tuple[int, str, float]]:
    """在 token 序列中定位关键词（子串匹配），返回 (idx, token, V)。"""
    kw = keyword.lower()
    hits: List[tuple[int, str, float]] = []
    for i, (tok, v) in enumerate(zip(tokens, valences)):
        nt = _norm_token(tok)
        if not nt:
            continue
        if nt == kw or kw in nt or nt in kw:
            hits.append((i, tok, v))
    if not hits:
        return None
    # 优先精确匹配
    for h in hits:
        if _norm_token(h[1]) == kw:
            return h
    return hits[0]


def analyze_text(model_loader, projector, text: str) -> dict:
    v_sent, coords, tokens = project_sentence_valence(
        model_loader, projector, text, aggregation="last"
    )
    return {
        "sentence_v": v_sent,
        "tokens": tokens,
        "token_v": coords["valence"],
    }


def main():
    from src.model_loader import CausalModelLoader  # noqa: E402

    print("=" * 80)
    print("B1 context test: 10 triplets × sentence V + keyword token V")
    print("Model: Qwen2.5-0.5B | Axes/scaler: E1 en/pole_50 | Forward-only")
    print("=" * 80)

    loader = CausalModelLoader("qwen", device="cpu")
    loader.load()
    projector, _ = load_e1_projector(loader)

    sent_rows: List[dict] = []
    kw_rows: List[dict] = []

    for item in TRIPLETS:
        gid = item["id"]
        for label in ("sarcastic", "positive", "negative"):
            text = item[label]
            res = analyze_text(loader, projector, text)
            sent_rows.append({
                "group": gid,
                "topic": item["topic"],
                "type": label,
                "sentence_v": round(res["sentence_v"], 4),
                "text": text,
            })

        for kw in item["keywords"]:
            row = {"group": gid, "topic": item["topic"], "keyword": kw}
            for label in ("sarcastic", "positive", "negative"):
                text = item[label]
                res = analyze_text(loader, projector, text)
                hit = find_keyword_valence(res["tokens"], res["token_v"], kw)
                if hit:
                    _, tok, v = hit
                    row[f"v_{label}"] = round(v, 4)
                    row[f"tok_{label}"] = tok
                else:
                    row[f"v_{label}"] = None
                    row[f"tok_{label}"] = None
            kw_rows.append(row)

    print("\n## Sentence-level V (last token)\n")
    print(f"{'G':>2}  {'Type':10s}  {'V_sent':>7s}  Topic")
    print("-" * 80)
    for r in sent_rows:
        print(f"{r['group']:2d}  {r['type']:10s}  {r['sentence_v']:7.3f}  {r['topic']}")

    # per-group summary
    print("\n## Per-group summary (S=sarcastic, P=positive, N=negative)\n")
    print(f"{'G':>2}  {'Topic':14s}  {'V_S':>7s}  {'V_P':>7s}  {'V_N':>7s}  {'S-P':>7s}  {'S-N':>7s}  {'P-N':>7s}")
    print("-" * 80)
    for gid in range(1, 11):
        grp = [r for r in sent_rows if r["group"] == gid]
        vs = {r["type"]: r["sentence_v"] for r in grp}
        topic = grp[0]["topic"]
        print(
            f"{gid:2d}  {topic:14s}  {vs['sarcastic']:7.3f}  {vs['positive']:7.3f}  "
            f"{vs['negative']:7.3f}  {vs['sarcastic'] - vs['positive']:7.3f}  "
            f"{vs['sarcastic'] - vs['negative']:7.3f}  {vs['positive'] - vs['negative']:7.3f}"
        )

    print("\n## Keyword token-level V\n")
    print(
        f"{'G':>2}  {'Keyword':12s}  {'V_sarc':>8s}  {'V_pos':>8s}  {'V_neg':>8s}  "
        f"{'Δ(S-P)':>8s}  {'Δ(S-N)':>8s}  Topic"
    )
    print("-" * 80)
    for r in kw_rows:
        vs = r.get("v_sarcastic")
        vp = r.get("v_positive")
        vn = r.get("v_negative")

        def _fmt(x):
            return f"{x:8.3f}" if x is not None else "     N/A"

        def _delta(a, b):
            if a is None or b is None:
                return "     N/A"
            return f"{a - b:8.3f}"

        print(
            f"{r['group']:2d}  {r['keyword']:12s}  {_fmt(vs)}  {_fmt(vp)}  {_fmt(vn)}  "
            f"{_delta(vs, vp)}  {_delta(vs, vn)}  {r['topic']}"
        )

    # aggregate stats for shared keywords (appear in both sarcastic and positive)
    shared = [
        r for r in kw_rows
        if r.get("v_sarcastic") is not None and r.get("v_positive") is not None
    ]
    if shared:
        diffs_sp = [r["v_sarcastic"] - r["v_positive"] for r in shared]
        mean_sp = sum(diffs_sp) / len(diffs_sp)
        print(f"\nShared keyword mean Δ(S-P): {mean_sp:.3f}  (n={len(shared)})")

    neg_kw = [r for r in kw_rows if r.get("v_sarcastic") is not None and r.get("v_negative") is not None]
    if neg_kw:
        # only where same keyword appears in both (rare) — skip
        pass

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
