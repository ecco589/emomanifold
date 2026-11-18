"""
RAL 补充：Llama-3.2-3B on Modal L4
  1) EN：复用 NIPS pole_50 极词 + scaler，重建轴 → A3 EN + B1 三元组
  2) ZH：按 Llama 分词器筛单 token 高极值情绪词 → 构轴 + 拟合 scaler → A3 ZH

前置（本机）:
  modal token new          # 若 `modal secret list` 报 Token not found
  # Modal secret 名 huggingface，内含 HF_TOKEN（Llama gated）

运行（仓库根目录）:
  python -m modal run RAL/experiment/modal/ral_llama32_a3_b1.py
  python -m modal run RAL/experiment/modal/ral_llama32_a3_b1.py --skip-b1   # 只跑 A3
  python -m modal run RAL/experiment/modal/ral_llama32_a3_b1.py --skip-zh   # 跳过中文构轴

结果落盘:
  RAL/experiment/A/A3/result/llama32_3b/
  RAL/experiment/B/B1/result/b1_full_metrics_llama32_3b.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


def _find_repo() -> Path:
    """Local checkout root. On Modal the entry script is mounted flat under /root/."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "RAL" / "experiment" / "modal").is_dir() and (p / "src").is_dir():
            return p
    return Path("/root")


REPO = _find_repo()
NIPS_EN = REPO / "RAL/experiment/A/A3/data/llama32_3b/en/pole_50"
A1_CODE = REPO / "RAL/experiment/A/A1/code"
A3_CODE = REPO / "RAL/experiment/A/A3/code"
A3_DATA = REPO / "RAL/experiment/A/A3/data"
B1_DATA = REPO / "RAL/experiment/B/B1/data"
A1_RESULT = REPO / "RAL/experiment/A/A1/result"

HF_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_KEY = "llama32_3b"
POLE_N = 50

# Image mounts are only resolved when building from the local repo.
_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "huggingface_hub",
    "accelerate",
    "scipy",
    "numpy",
    "tqdm",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

if (REPO / "src").is_dir():
    image = (
        _image.add_local_dir(REPO / "src", remote_path="/root/src")
        .add_local_file(REPO / "config.py", remote_path="/root/config.py")
        .add_local_dir(
            REPO / "data" / "NRC-VAD-Lexicon", remote_path="/root/data/NRC-VAD-Lexicon"
        )
        .add_local_dir(A1_CODE, remote_path="/root/ral_a1_code")
        .add_local_dir(A3_CODE, remote_path="/root/ral_a3_code")
        .add_local_dir(A3_DATA, remote_path="/root/ral_a3_data")
        .add_local_dir(B1_DATA, remote_path="/root/ral_b1_data")
        .add_local_dir(A1_RESULT, remote_path="/root/ral_a1_result")
        .add_local_dir(NIPS_EN, remote_path="/root/llama_en_pole50")
    )
else:
    image = _image

model_volume = modal.Volume.from_name("emotion-manifold-models", create_if_missing=True)
MODEL_CACHE = "/cache/models"

app = modal.App("ral-llama32-a3-b1", image=image)


class _LoaderAdapter:
    """Minimal adapter matching EmotionProjector / E1AdaptiveScaler expectations."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if hasattr(model, "get_input_embeddings"):
            self.output_embedding = model.get_input_embeddings().weight.data
        else:
            self.output_embedding = model.lm_head.weight.data

    def get_output_embedding_matrix(self):
        return self.output_embedding

    def get_hidden_states(self, input_ids, return_dict=True):
        import torch

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        if return_dict:
            return {"hidden_states": hidden, "last_hidden_state": hidden}
        return hidden

    def tokenize(self, text, return_tokens=False):
        raw = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(raw)
        if return_tokens:
            readable = []
            for tid in ids:
                try:
                    readable.append(self.tokenizer.decode([tid], skip_special_tokens=False).strip() or f"[{tid}]")
                except Exception:
                    readable.append(f"[{tid}]")
            return ids, readable
        return ids


def _load_llama(device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(
        HF_ID, cache_dir=MODEL_CACHE, token=token, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID,
        cache_dir=MODEL_CACHE,
        token=token,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return _LoaderAdapter(model, tok, device if device != "cuda" else str(model.device))


def _axes_from_lex(loader, lex_path: str):
    import sys

    sys.path.insert(0, "/root/ral_a1_code")
    from e1_core import extract_axes_from_embedding

    emb = loader.get_output_embedding_matrix()
    return extract_axes_from_embedding(emb, loader.tokenizer, lex_path, device="cpu")


def _run_a3_condition(name: str, loader, lex_path: Path, scaler_params: dict, clusters: list):
    import sys

    sys.path.insert(0, "/root/ral_a3_code")
    from e3_core import evaluate_clusters, orthogonality, summarize_condition

    axes = _axes_from_lex(loader, str(lex_path))
    orth = orthogonality(axes)
    emb = loader.get_output_embedding_matrix().float().cpu()
    cluster_cos, skipped = evaluate_clusters(emb, loader.tokenizer, clusters, axes, scaler_params)
    return summarize_condition(name, str(lex_path), scaler_params, orth, cluster_cos, skipped), axes


def _soft_pole_ok(pole: str, vad: dict, thresh: float) -> bool:
    """Relax dominant-dim rule; only require signed extremity on the target axis."""
    v, a, d = vad["valence"], vad["arousal"], vad["dominance"]
    if pole == "valence_positive":
        return v > thresh
    if pole == "valence_negative":
        return v < -thresh
    if pole == "arousal_high":
        return a > thresh
    if pole == "arousal_low":
        return a < -thresh
    if pole == "dominance_high":
        return d > thresh
    if pole == "dominance_low":
        return d < -thresh
    return False


def _select_zh_groups_for_llama(tokenizer, nrc: dict, priority: dict, pole_n: int) -> tuple[dict, dict]:
    """
    Llama 对中文多字词常切成多 token；严格 dominant-dim + 单 token 装不满。
    策略：单 token 硬约束；阈值 0.45→0.30→0.20→0.12 递降；允许跨极复用作最后手段。
    """
    import sys

    sys.path.insert(0, "/root/ral_a1_code")
    from b1_lexicon import POLE_ORDER, is_single_token, extremity_key, _dedupe_preserve

    # Prefer shorter CJK (more often single-token under Llama BPE)
    EXTRA_SEEDS = {
        "valence_positive": ["喜", "乐", "爱", "甜", "美", "暖", "福", "吉", "好", "棒"],
        "valence_negative": ["悲", "痛", "苦", "恨", "哀", "惨", "怒", "惧", "惧", "恶"],
        "arousal_high": ["怒", "狂", "慌", "惊", "急", "躁", "烈", "爆", "激", "吓"],
        "arousal_low": ["静", "闲", "懒", "倦", "眠", "缓", "淡", "稳", "定", "宁"],
        "dominance_high": ["强", "威", "霸", "傲", "胜", "王", "尊", "刚", "猛", "雄"],
        "dominance_low": ["弱", "卑", "怯", "怕", "羞", "软", "小", "屈", "奴", "残"],
    }

    merged_priority = {p: _dedupe_preserve(list(priority.get(p, [])) + EXTRA_SEEDS.get(p, [])) for p in POLE_ORDER}

    # Precompute single-token NRC pool
    single_pool = [w for w in nrc if is_single_token(w, tokenizer)]
    diag = {
        "nrc_zh_n": len(nrc),
        "single_token_nrc_n": len(single_pool),
        "thresholds_tried": [],
        "filled_with": {},
    }

    def try_fill(thresh: float, allow_reuse: bool) -> dict | None:
        used: set[str] = set()
        groups = {p: [] for p in POLE_ORDER}

        def add(pole: str, w: str) -> bool:
            if w not in nrc or not _soft_pole_ok(pole, nrc[w], thresh):
                return False
            if not is_single_token(w, tokenizer):
                return False
            if w in groups[pole]:
                return False
            if not allow_reuse and w in used:
                return False
            groups[pole].append(w)
            used.add(w)
            return True

        for pole in POLE_ORDER:
            for w in merged_priority[pole]:
                if len(groups[pole]) >= pole_n:
                    break
                add(pole, w)

        for pole in POLE_ORDER:
            if len(groups[pole]) >= pole_n:
                continue
            ranked = sorted(
                (w for w in single_pool if _soft_pole_ok(pole, nrc[w], thresh)),
                key=lambda w: (-extremity_key(pole, nrc[w]), len(w), w),
            )
            for w in ranked:
                if len(groups[pole]) >= pole_n:
                    break
                add(pole, w)

        if all(len(groups[p]) >= pole_n for p in POLE_ORDER):
            return {p: groups[p][:pole_n] for p in POLE_ORDER}
        return None

    word_groups = None
    for thresh in (0.45, 0.35, 0.30, 0.22, 0.15, 0.10):
        for reuse in (False, True):
            diag["thresholds_tried"].append({"thresh": thresh, "reuse": reuse})
            got = try_fill(thresh, reuse)
            if got is not None:
                word_groups = got
                diag["filled_with"] = {"thresh": thresh, "allow_cross_pole_reuse": reuse}
                break
        if word_groups is not None:
            break

    if word_groups is None:
        # Last resort: take whatever we can at thresh=0.10 with reuse, pad by length report
        got = try_fill(0.08, True)
        counts = {p: len(got[p]) if got else 0 for p in POLE_ORDER}
        raise RuntimeError(
            f"ZH poles still short under Llama single-token filter: {counts}. "
            f"single_token_nrc={diag['single_token_nrc_n']}"
        )

    # per-pole diagnostics
    from b1_lexicon import pole_predicate

    for pole in POLE_ORDER:
        seeds = merged_priority[pole]
        single = [w for w in seeds if is_single_token(w, tokenizer)]
        strict = [w for w in single if w in nrc and pole_predicate(pole, nrc[w])]
        diag[pole] = {
            "priority_n": len(seeds),
            "single_token_n": len(single),
            "strict_predicate_ok_n": len(strict),
            "final_n": len(word_groups[pole]),
            "final_sample": word_groups[pole][:12],
            "mean_char_len": round(
                sum(len(w) for w in word_groups[pole]) / max(len(word_groups[pole]), 1), 2
            ),
        }
    return word_groups, diag


def _build_zh_poles(loader, pole_n: int = POLE_N) -> dict:
    """Careful ZH poles: priority + NRC → Llama single-token soft fill → scaler."""
    import sys

    sys.path.insert(0, "/root/ral_a1_code")
    from b1_lexicon import (
        extend_priority_from_nrc,
        load_language_nrc,
        build_lexicon_payload,
        build_corpus_lines,
        PRIORITY_ZH,
    )
    from e1_core import (
        E1AdaptiveScaler,
        extract_axes_from_embedding,
        compute_spearman,
    )
    from src.emotion_projector import EmotionProjector

    project_root = Path("/root")
    nrc = load_language_nrc("zh", project_root)
    priority = extend_priority_from_nrc(PRIORITY_ZH, nrc, top_k=400)

    word_groups, diag = _select_zh_groups_for_llama(
        loader.tokenizer, nrc, priority, pole_n
    )
    print("ZH fill:", json.dumps(diag.get("filled_with"), ensure_ascii=False), flush=True)
    print(
        "ZH counts:",
        {k: diag[k]["final_n"] for k in word_groups},
        "mean_char_len:",
        {k: diag[k]["mean_char_len"] for k in word_groups},
        flush=True,
    )

    lex_payload = build_lexicon_payload(word_groups, nrc, "zh")
    corpus = build_corpus_lines(word_groups, "zh")

    out = {
        "diag_priority_vs_tokenizer": diag,
        "word_groups": word_groups,
        "lex_payload": lex_payload,
        "corpus": corpus,
    }

    emb = loader.get_output_embedding_matrix()
    lex_path = Path("/tmp/zh_pole_lexicon.json")
    lex_path.write_text(json.dumps(lex_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    axes = extract_axes_from_embedding(emb, loader.tokenizer, str(lex_path), device="cpu")
    projector_raw = EmotionProjector(axes, scaler=None, device="cpu")
    scaler = E1AdaptiveScaler(lex_payload, device="cpu")
    scaler.train(corpus, loader, projector_raw, verbose=False)
    nonneg = scaler.enforce_nonnegative_scales()
    scaler_params = {
        d: {"scale": float(scaler.scales[d]), "bias": float(scaler.biases[d])}
        for d in axes
    }

    holdout_path = Path("/root/ral_a1_result/holdout_zh.json")
    spearman = None
    spearman_meta = None
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        spearman, spearman_meta = compute_spearman(
            emb, axes, scaler_params, holdout, loader.tokenizer
        )

    out.update(
        {
            "scaler": scaler_params,
            "scaler_nonneg_adjustment": nonneg,
            "spearman": spearman,
            "spearman_meta": spearman_meta,
            "lex_path_tmp": str(lex_path),
        }
    )
    return out


def _run_b1(loader, axes, scaler_params, lex_payload: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/ral_a1_code")
    from src.emotion_projector import EmotionProjector
    from src.adaptive_scaler import AdaptiveScaler

    scaler = AdaptiveScaler(lex_payload, device="cpu")
    for dim, params in scaler_params.items():
        scaler.scales[dim] = float(params["scale"])
        scaler.biases[dim] = float(params["bias"])
    scaler.is_trained = True
    projector = EmotionProjector(axes, scaler=scaler, device="cpu")

    triplets_path = Path("/root/ral_b1_data/context_triplets_en.json")
    dataset = json.loads(triplets_path.read_text(encoding="utf-8"))
    items = dataset["items"]
    labels = ("sarcastic", "positive", "negative")
    dims = ("valence", "arousal", "dominance")

    def project_text(text: str):
        coords, tokens = projector.project_text(loader, text, apply_scaling=True)
        return coords, tokens

    def find_kw(tokens, vals, kw: str):
        kw_l = kw.lower()
        for i, t in enumerate(tokens):
            tl = t.lower().strip()
            if tl == kw_l or tl.replace("Ġ", "") == kw_l or kw_l in tl:
                return i, t
        # join-based fallback
        joined = "".join(tokens).lower()
        return None

    sent_rows = []
    kw_rows = []
    for item in items:
        gid = item["id"]
        for label in labels:
            coords, tokens = project_text(item[label])
            sent_rows.append(
                {
                    "id": gid,
                    "type": label,
                    "sentence_v": round(coords["valence"][-1], 4) if coords["valence"] else None,
                    "intent": item.get("intent", ""),
                }
            )
        for kw in item.get("keywords", []):
            row = {"id": gid, "keyword": kw, "intent": item.get("intent", "")}
            for label in labels:
                coords, tokens = project_text(item[label])
                hit = None
                kw_l = kw.lower()
                for i, t in enumerate(tokens):
                    tl = t.lower().replace("Ġ", "").replace("▁", "").strip()
                    if tl == kw_l or kw_l.startswith(tl) or tl.startswith(kw_l):
                        hit = (i, t)
                        break
                for dim in dims:
                    row[f"{dim}_{label}"] = (
                        round(coords[dim][hit[0]], 4) if hit and coords.get(dim) else None
                    )
                row[f"tok_{label}"] = hit[1] if hit else None
            kw_rows.append(row)

    def group(label):
        return [r["sentence_v"] for r in sent_rows if r["type"] == label and r["sentence_v"] is not None]

    vs, vp, vn = group("sarcastic"), group("positive"), group("negative")
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    by_id = {}
    for r in sent_rows:
        by_id.setdefault(r["id"], {})[r["type"]] = r["sentence_v"]
    s_minus_p = [
        by_id[i]["sarcastic"] - by_id[i]["positive"]
        for i in by_id
        if by_id[i].get("sarcastic") is not None and by_id[i].get("positive") is not None
    ]
    s_minus_n = [
        by_id[i]["sarcastic"] - by_id[i]["negative"]
        for i in by_id
        if by_id[i].get("sarcastic") is not None and by_id[i].get("negative") is not None
    ]
    primary = {}
    for item in items:
        if item.get("keywords"):
            primary[item["id"]] = item["keywords"][0]
    primary_rows = [r for r in kw_rows if r["keyword"] == primary.get(r["id"])]
    delta_primary = [
        r["valence_sarcastic"] - r["valence_positive"]
        for r in primary_rows
        if r.get("valence_sarcastic") is not None and r.get("valence_positive") is not None
    ]
    delta_all = [
        r["valence_sarcastic"] - r["valence_positive"]
        for r in kw_rows
        if r.get("valence_sarcastic") is not None and r.get("valence_positive") is not None
    ]

    summary = {
        "experiment": "B1_full",
        "n_triplets": len(items),
        "model": HF_ID,
        "axes": "NIPS llama32_3b en/pole_50",
        "sentence_level": {
            "mean_v_sarcastic": round(mean(vs), 4) if mean(vs) is not None else None,
            "mean_v_positive": round(mean(vp), 4) if mean(vp) is not None else None,
            "mean_v_negative": round(mean(vn), 4) if mean(vn) is not None else None,
            "mean_delta_s_minus_n": round(mean(s_minus_n), 4) if mean(s_minus_n) is not None else None,
            "mean_delta_s_minus_p": round(mean(s_minus_p), 4) if mean(s_minus_p) is not None else None,
            "fraction_abs_delta_sn_lt_0.15": round(
                sum(1 for d in s_minus_n if abs(d) < 0.15) / max(len(s_minus_n), 1), 4
            ),
        },
        "keyword_level_valence": {
            "n_keyword_rows": len(kw_rows),
            "n_shared_s_p": len(delta_all),
            "mean_delta_v_s_minus_p_all_kw": round(mean(delta_all), 4) if delta_all else None,
            "mean_delta_v_s_minus_p_primary_kw": round(mean(delta_primary), 4) if delta_primary else None,
        },
        "scaler": scaler_params,
    }
    return {"summary": summary, "sentence": sent_rows, "keywords": kw_rows}


@app.function(
    gpu="L4",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={MODEL_CACHE: model_volume},
    memory=32768,
)
def run_remote(skip_b1: bool = False, skip_zh: bool = False) -> dict:
    import sys
    import time

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/ral_a1_code")
    sys.path.insert(0, "/root/ral_a3_code")

    from e3_core import load_json, random_baseline_clusters

    t0 = time.time()
    print(f"Loading {HF_ID} …", flush=True)
    loader = _load_llama("cuda")
    model_volume.commit()

    clusters = load_json(Path("/root/ral_a3_data/clusters.json"))["clusters"]
    print("Random 3D baseline …", flush=True)
    emb = loader.get_output_embedding_matrix().float().cpu()
    baseline = random_baseline_clusters(
        emb, loader.tokenizer, clusters, n_trials=100, seed=42
    )

    # ---- EN from NIPS ----
    en_lex = Path("/root/llama_en_pole50/pole_lexicon.json")
    en_report = json.loads(Path("/root/llama_en_pole50/report.json").read_text())
    en_scaler = en_report["scaler"]
    print("A3 EN …", flush=True)
    en_cond, en_axes = _run_a3_condition("en_only", loader, en_lex, en_scaler, clusters)
    print(f"  EN cluster mean={en_cond['cluster_cos']['mean']} orth={en_cond['orthogonality']}", flush=True)

    zh_pack = None
    zh_cond = None
    if not skip_zh:
        print("Building ZH poles under Llama tokenizer …", flush=True)
        zh_pack = _build_zh_poles(loader, POLE_N)
        print("ZH diag:", json.dumps(zh_pack["diag_priority_vs_tokenizer"], ensure_ascii=False), flush=True)
        print("ZH spearman:", zh_pack.get("spearman"), flush=True)
        zh_lex_path = Path("/tmp/zh_pole_lexicon.json")
        print("A3 ZH …", flush=True)
        zh_cond, _ = _run_a3_condition(
            "zh_only", loader, zh_lex_path, zh_pack["scaler"], clusters
        )
        print(f"  ZH cluster mean={zh_cond['cluster_cos']['mean']} orth={zh_cond['orthogonality']}", flush=True)

    b1 = None
    if not skip_b1:
        print("B1 full n=200 …", flush=True)
        en_lex_payload = json.loads(en_lex.read_text(encoding="utf-8"))
        # AdaptiveScaler only needs a words dict at init; scales are injected.
        # NIPS poles may contain words absent from local BipolarScale NRC.
        if "words" not in en_lex_payload or not en_lex_payload["words"]:
            sys.path.insert(0, "/root/ral_a1_code")
            from b1_lexicon import load_language_nrc

            nrc_en = load_language_nrc("en", Path("/root"))
            words = {}
            for ws in en_lex_payload["word_groups"].values():
                for w in ws:
                    if w in nrc_en:
                        words[w] = dict(nrc_en[w])
                    else:
                        words[w] = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            en_lex_payload = {
                "description": "Llama EN poles (NIPS) for B1 projector",
                "language": "en",
                "normalized_to_minus_one_one": True,
                "word_groups": en_lex_payload["word_groups"],
                "words": words,
            }
        b1 = _run_b1(loader, en_axes, en_scaler, en_lex_payload)
        print("B1 summary:", json.dumps(b1["summary"], ensure_ascii=False), flush=True)

    elapsed = time.time() - t0
    payload = {
        "model": HF_ID,
        "model_key": MODEL_KEY,
        "pole_n": POLE_N,
        "elapsed_seconds": elapsed,
        "random_baseline_cluster_cos": baseline,
        "a3": {
            "en_only": en_cond,
            "zh_only": zh_cond,
        },
        "zh_build": None
        if zh_pack is None
        else {
            "diag_priority_vs_tokenizer": zh_pack["diag_priority_vs_tokenizer"],
            "word_groups": zh_pack["word_groups"],
            "scaler": zh_pack["scaler"],
            "scaler_nonneg_adjustment": zh_pack["scaler_nonneg_adjustment"],
            "spearman": zh_pack["spearman"],
            "lex_payload": zh_pack["lex_payload"],
            "corpus": zh_pack["corpus"],
        },
        "b1": b1,
    }
    return payload


@app.local_entrypoint()
def main(skip_b1: bool = False, skip_zh: bool = False):
    print("Remote L4 job starting …")
    result = run_remote.remote(skip_b1=skip_b1, skip_zh=skip_zh)

    a3_out = REPO / "RAL/experiment/A/A3/result/llama32_3b"
    a3_out.mkdir(parents=True, exist_ok=True)
    (a3_out / "a3_metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E3",
                "model": result["model"],
                "pole_n": POLE_N,
                "random_baseline_cluster_cos": result["random_baseline_cluster_cos"],
                "conditions": {
                    k: v
                    for k, v in result["a3"].items()
                    if v is not None
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # EN already present; write ZH artifacts
    if result.get("zh_build"):
        zh_dir = a3_out / "zh" / "pole_50"
        zh_dir.mkdir(parents=True, exist_ok=True)
        zb = result["zh_build"]
        (zh_dir / "pole_lexicon.json").write_text(
            json.dumps(zb["lex_payload"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (zh_dir / "scaler.json").write_text(
            json.dumps(zb["scaler"], indent=2), encoding="utf-8"
        )
        (zh_dir / "training_corpus.txt").write_text(
            "\n".join(zb["corpus"]) + "\n", encoding="utf-8"
        )
        (zh_dir / "report.json").write_text(
            json.dumps(
                {
                    "experiment": "E1",
                    "language": "zh",
                    "model": MODEL_KEY,
                    "model_id": HF_ID,
                    "pole_n": POLE_N,
                    "word_groups": zb["word_groups"],
                    "scaler": zb["scaler"],
                    "scaler_nonneg_adjustment": zb["scaler_nonneg_adjustment"],
                    "spearman": zb["spearman"],
                    "diag_priority_vs_tokenizer": zb["diag_priority_vs_tokenizer"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (zh_dir / "diag_tokenizer.json").write_text(
            json.dumps(zb["diag_priority_vs_tokenizer"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # also mirror under A1-style path for symmetry
        a1_zh = REPO / "RAL/experiment/A/A1/result/llama32_3b/zh/pole_50"
        a1_zh.mkdir(parents=True, exist_ok=True)
        for name in ("pole_lexicon.json", "scaler.json", "report.json", "training_corpus.txt"):
            (a1_zh / name).write_text((zh_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    if result.get("b1"):
        b1_path = REPO / "RAL/experiment/B/B1/result/b1_full_metrics_llama32_3b.json"
        b1_path.parent.mkdir(parents=True, exist_ok=True)
        b1_path.write_text(json.dumps(result["b1"], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {b1_path}")

    print(f"Wrote {a3_out / 'a3_metrics.json'}")
    print(f"Elapsed remote: {result['elapsed_seconds']:.1f}s")
    if result["a3"].get("en_only"):
        print("A3 EN:", result["a3"]["en_only"]["orthogonality"], result["a3"]["en_only"]["cluster_cos"])
    if result["a3"].get("zh_only"):
        print("A3 ZH:", result["a3"]["zh_only"]["orthogonality"], result["a3"]["zh_only"]["cluster_cos"])
    if result.get("b1"):
        print("B1:", result["b1"]["summary"]["keyword_level_valence"])
