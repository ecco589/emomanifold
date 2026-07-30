"""
Mishra-style turn-level discrete emotion baseline (Frontiers 2023 prompt).

Usage (from repo root):
  python RAL/experiment/C/C1/code/baselines/run_mishra.py --model qwen
  python RAL/experiment/C/C1/code/baselines/run_mishra.py --model gpt   # needs OPENAI_API_KEY

Same model via OpenRouter (recommended if OpenAI billing fails):
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1
  export OPENAI_API_KEY=sk-or-...
  python RAL/experiment/C/C1/code/baselines/run_mishra.py --model gpt \\
      --gpt-model openai/gpt-4o-mini

Then:
  python RAL/experiment/C/C1/code/baselines/weights_from_label.py --input .../mishra_qwen.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STIMULI = ROOT / "RAL/experiment/C/C1/data/stimuli.csv"
PROMPT_FILE = Path(__file__).resolve().parent / "mishra_prompt.txt"
OUT_DIR = Path(__file__).resolve().parents[2] / "result" / "label_precompute"

# Paper IDs 1-6 → our 7-way names (no disgust in Mishra)
ID_TO_NAME = {
    1: "happiness",
    2: "sadness",
    3: "fear",
    4: "anger",
    5: "surprise",
    6: "neutral",
}


def load_stimuli(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_prompt(prefix: str, robot_utterance: str, human: str = "Please react to the following.") -> str:
    return (
        f"{prefix.rstrip()}\n\n"
        f"Human: {human}\n"
        f"Robot: {robot_utterance}\n"
        f"Robot: ("
    )


def parse_emotion_id(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"[1-6]", text.strip())
    return int(m.group(0)) if m else None


def run_gpt(prompt: str, model: str, base_url: str | None = None) -> tuple[str, float]:
    from openai import OpenAI

    # OPENAI_BASE_URL / OPENAI_API_KEY are also read by the SDK; explicit base_url
    # lets us log the gateway (OpenRouter / Azure / relay) without changing the model.
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    t0 = time.perf_counter()
    # Chat models: ask for a single digit; keep completion-like instruction.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt + "\n\nReply with only one digit 1-6.",
            }
        ],
        max_tokens=8,
        temperature=0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = (resp.choices[0].message.content or "").strip()
    return raw, latency_ms


def run_qwen(prompt: str, device: str) -> tuple[str, float]:
    """Local Qwen: next-token argmax among digit strings 1-6 (completion-style classifier)."""
    import torch
    from src.model_loader import CausalModelLoader

    if not hasattr(run_qwen, "_loader"):
        loader = CausalModelLoader("qwen", device=device)
        loader.load()
        run_qwen._loader = loader
        # Candidate token ids for each class id (first-token completion after "Robot: (").
        tok = loader.tokenizer
        cand: dict[int, list[int]] = {}
        for d in range(1, 7):
            ids = set()
            for s in (str(d), f" {d}", f"{d})", f" {d})"):
                enc = tok.encode(s, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            cand[d] = sorted(ids)
        run_qwen._digit_cands = cand

    loader = run_qwen._loader
    tok = loader.tokenizer
    model = loader.model
    digit_cands: dict[int, list[int]] = run_qwen._digit_cands

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(loader.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1]  # vocab
        best_d, best_logit = None, float("-inf")
        for d, ids in digit_cands.items():
            if not ids:
                continue
            score = max(float(logits[i]) for i in ids)
            if score > best_logit:
                best_logit, best_d = score, d
    latency_ms = (time.perf_counter() - t0) * 1000
    if best_d is None:
        return "", latency_ms
    return str(best_d), latency_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt", "qwen"], required=True)
    ap.add_argument("--gpt-model", default=os.environ.get("MISHRA_GPT_MODEL", "gpt-4o-mini"))
    ap.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible gateway, e.g. https://openrouter.ai/api/v1",
    )
    ap.add_argument("--device", default=os.environ.get("MISHRA_DEVICE", "cpu"))
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per sentence (for latency stats)")
    args = ap.parse_args()

    if args.model == "gpt" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "缺少 OPENAI_API_KEY。OpenRouter: export OPENAI_API_KEY=sk-or-... "
            "且 OPENAI_BASE_URL=https://openrouter.ai/api/v1"
        )

    # OpenRouter uses openai/gpt-4o-mini; keep underlying model = GPT-4o-mini.
    gpt_model = args.gpt_model
    base_url = (args.base_url or "").strip() or None
    if base_url and "openrouter.ai" in base_url and "/" not in gpt_model:
        gpt_model = f"openai/{gpt_model}"

    prefix = PROMPT_FILE.read_text(encoding="utf-8")
    stimuli = load_stimuli(args.stimuli)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mishra_{args.model}.csv"

    fieldnames = [
        "id",
        "category",
        "user_prompt",
        "repeat",
        "emotion_id",
        "emotion_name",
        "raw_output",
        "latency_ms",
        "model",
        "gateway",
        "error",
    ]
    rows_out = []

    print(f"Stimuli: {args.stimuli} ({len(stimuli)} sentences)")
    if args.model == "gpt":
        print(f"Backend: gpt model={gpt_model} gateway={base_url or 'api.openai.com'}")
    else:
        print(f"Backend: qwen device={args.device}")

    for row in stimuli:
        sid = row["id"]
        text = row["user_prompt"]
        prompt = build_prompt(prefix, text)
        for r in range(1, args.repeats + 1):
            err = ""
            raw = ""
            eid = None
            name = ""
            latency = float("nan")
            try:
                if args.model == "gpt":
                    raw, latency = run_gpt(prompt, gpt_model, base_url=base_url)
                    model_tag = gpt_model
                else:
                    raw, latency = run_qwen(prompt, args.device)
                    model_tag = "Qwen2.5-0.5B"
                eid = parse_emotion_id(raw)
                if eid is None:
                    err = "parse_failed"
                else:
                    name = ID_TO_NAME[eid]
            except Exception as e:
                err = str(e)
                model_tag = gpt_model if args.model == "gpt" else "Qwen2.5-0.5B"

            rec = {
                "id": sid,
                "category": row.get("category", ""),
                "user_prompt": text,
                "repeat": r,
                "emotion_id": eid if eid is not None else "",
                "emotion_name": name,
                "raw_output": raw.replace("\n", " ").strip(),
                "latency_ms": f"{latency:.2f}" if latency == latency else "",
                "model": model_tag,
                "gateway": base_url or ("openai" if args.model == "gpt" else ""),
                "error": err,
            }
            rows_out.append(rec)
            print(
                f"[{sid} r{r}] id={eid} name={name or '-'} "
                f"latency={latency:.1f}ms raw={raw!r} {err}"
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
