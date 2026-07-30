"""
Xpress conversational emotion+intensity baseline (prompt from official realtime_server.js).

Usage (repo root):
  python RAL/experiment/C/C1/code/baselines/run_xpress_conv.py --model gpt
  python RAL/experiment/C/C1/code/baselines/run_xpress_conv.py --model qwen --device cpu

Then:
  python RAL/experiment/C/C1/code/baselines/weights_from_label.py \\
      --input RAL/experiment/C/C1/result/label_precompute/xpress_gpt.csv --intensity-col intensity
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STIMULI = ROOT / "RAL/experiment/C/C1/data/stimuli.csv"
PROMPT_FILE = Path(__file__).resolve().parent / "xpress_conv_prompt.txt"
LABEL_MAP_FILE = Path(__file__).resolve().parent / "xpress_label_map.json"
OUT_DIR = Path(__file__).resolve().parents[2] / "result" / "label_precompute"

# Turn sentences: split into before/after contrast for chunk-style labeling.
TURN_SPLITS = {
    "013": (
        "I was worried at first",
        "but the result was much better than I expected.",
    ),
    "014": (
        "I was really looking forward to it",
        "but ended up so disappointed.",
    ),
}


def load_stimuli(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_user_message(
    previous: str,
    current: str,
    current_expression: str = "neutral",
    time_since: str = "5 seconds",
) -> str:
    prev = previous if previous else "{}"
    return (
        f"previous_chunks: {{{prev}}}, "
        f"current_chunk: {{{current}}}, "
        f"current-expression: {current_expression}, "
        f"time-since-expression-change: {time_since}"
    )


def parse_reaction(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    raw = text.strip()
    # Strip markdown fences if any
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.M).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate single quotes / unquoted-ish dumps from examples
        m_e = re.search(r"['\"]?emotion['\"]?\s*[:=]\s*['\"]?([a-zA-Z\-]+)['\"]?", raw, re.I)
        m_i = re.search(r"['\"]?intensity['\"]?\s*[:=]\s*['\"]?(high|low)['\"]?", raw, re.I)
        if not m_e:
            return None, None
        return m_e.group(1).lower(), (m_i.group(1).lower() if m_i else "high")
    emo = str(obj.get("emotion", "")).strip().lower()
    inten = str(obj.get("intensity", "")).strip().lower()
    if inten not in ("high", "low"):
        inten = "high" if inten else None
    return (emo or None), inten


def map_emotion(emo: str, label_map: dict[str, str]) -> str:
    if emo in label_map:
        return label_map[emo]
    # fuzzy: strip trailing 'ed'
    if emo.endswith("ed") and emo[:-2] in label_map:
        return label_map[emo[:-2]]
    return "neutral"


def segments_for_row(row: dict) -> list[str]:
    sid = row["id"]
    if sid in TURN_SPLITS:
        return list(TURN_SPLITS[sid])
    return [row["user_prompt"]]


def run_gpt(system: str, user: str, model: str, base_url: str | None) -> tuple[str, float]:
    from openai import OpenAI

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(timeout=120.0, **kwargs)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=64,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return (resp.choices[0].message.content or "").strip(), latency_ms


def run_qwen(system: str, user: str, device: str) -> tuple[str, float]:
    """Local Qwen completion; ask for compact JSON (base model is weak)."""
    import torch
    from src.model_loader import CausalModelLoader

    if not hasattr(run_qwen, "_loader"):
        loader = CausalModelLoader("qwen", device=device)
        loader.load()
        run_qwen._loader = loader
    loader = run_qwen._loader
    tok = loader.tokenizer
    model = loader.model

    prompt = (
        f"{system}\n\nINPUT: {user}\nOUTPUT: "
    )
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(loader.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip(), latency_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt", "qwen"], required=True)
    ap.add_argument("--gpt-model", default=os.environ.get("XPRESS_GPT_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--device", default=os.environ.get("XPRESS_DEVICE", "cpu"))
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    args = ap.parse_args()

    if args.model == "gpt" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY")

    system = PROMPT_FILE.read_text(encoding="utf-8").strip()
    label_map = json.loads(LABEL_MAP_FILE.read_text(encoding="utf-8"))
    stimuli = load_stimuli(args.stimuli)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"xpress_{args.model}.csv"

    gpt_model = args.gpt_model
    base_url = (args.base_url or "").strip() or None
    if base_url and "openrouter.ai" in base_url and "/" not in gpt_model:
        gpt_model = f"openai/{gpt_model}"

    fieldnames = [
        "id",
        "category",
        "segment",
        "segment_text",
        "user_prompt",
        "emotion",
        "intensity",
        "emotion_name",
        "raw_output",
        "latency_ms",
        "model",
        "gateway",
        "error",
    ]
    rows_out: list[dict] = []

    print(f"Stimuli: {args.stimuli} ({len(stimuli)})")
    if args.model == "gpt":
        print(f"Backend: gpt model={gpt_model} gateway={base_url or 'api.openai.com'}")
    else:
        print(f"Backend: qwen device={args.device}")

    for row in stimuli:
        sid = row["id"]
        segs = segments_for_row(row)
        prev = ""
        cur_expr = "neutral"
        for si, seg in enumerate(segs, start=1):
            user = build_user_message(prev, seg, cur_expr, "5 seconds")
            err = ""
            raw = ""
            emo = None
            inten = None
            mapped = ""
            latency = float("nan")
            try:
                if args.model == "gpt":
                    raw, latency = run_gpt(system, user, gpt_model, base_url)
                    model_tag = gpt_model
                else:
                    raw, latency = run_qwen(system, user, args.device)
                    model_tag = "Qwen2.5-0.5B"
                emo, inten = parse_reaction(raw)
                if emo is None or inten is None:
                    err = "parse_failed"
                else:
                    mapped = map_emotion(emo, label_map)
                    if emo != "no-change":
                        cur_expr = emo
            except Exception as e:
                err = str(e)
                model_tag = gpt_model if args.model == "gpt" else "Qwen2.5-0.5B"

            rec = {
                "id": sid,
                "category": row.get("category", ""),
                "segment": si,
                "segment_text": seg,
                "user_prompt": row["user_prompt"],
                "emotion": emo or "",
                "intensity": inten or "",
                "emotion_name": mapped,
                "raw_output": raw.replace("\n", " ").strip(),
                "latency_ms": f"{latency:.2f}" if latency == latency else "",
                "model": model_tag,
                "gateway": base_url or ("openai" if args.model == "gpt" else ""),
                "error": err,
            }
            rows_out.append(rec)
            print(
                f"[{sid} s{si}] emo={emo or '-'} inten={inten or '-'} "
                f"-> {mapped or '-'} {latency:.0f}ms {err}"
            )
            prev = (prev + " " + seg).strip() if prev else seg

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
