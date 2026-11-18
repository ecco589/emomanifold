#!/usr/bin/env python3
"""Time Xpress-Story's two original prompts, without speech synthesis.

The clock is the same one as Mishra and Xpress-Turn: the reply text is
already finished, and we time the calls that produce the first face
command. Xpress-Story needs both of its published prompts, in order:
segmentStory (temperature 0.1) then story2Emotion (temperature 0.3).
The user message is the reply with the word timestamps those prompts
require. TTS and whisper are not on this clock.

  python time_story_latency.py --backend gpt --key openai_key
  python time_story_latency.py --backend qwen

Writes one json per trial plus summary.csv. A finished trial is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "xpress_story_prompts"
HERE = Path(__file__).resolve().parent


def load_prompts(prompt_dir: Path) -> tuple[list, list]:
    segment = json.loads((prompt_dir / "segmentStory.json").read_text(encoding="utf-8"))
    emotion = json.loads((prompt_dir / "story2Emotion.json").read_text(encoding="utf-8"))
    return segment, emotion


def gpt_call(key: str, messages: list, temperature: float) -> tuple[str, float]:
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    dt = (time.perf_counter() - t0) * 1000.0
    return body["choices"][0]["message"]["content"], dt


def gpt_call_retry(key: str, messages: list, temperature: float) -> tuple[str, float]:
    last = None
    for attempt in range(4):
        try:
            return gpt_call(key, messages, temperature)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:300]
            last = RuntimeError(f"HTTP {e.code}: {err}")
            if e.code not in (429, 500, 502, 503) or attempt == 3:
                raise last
        except urllib.error.URLError as e:
            last = e
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    raise last


class _JsonReady:
    """Stop a local generation once a top-level JSON object has closed."""

    def __init__(self, tokenizer, prompt_len: int, limit: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.limit = limit

    def __call__(self, input_ids, scores, **kwargs):
        new = input_ids[0, self.prompt_len:]
        if len(new) >= self.limit:
            return True
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        start = text.find("{")
        if start < 0:
            return False
        depth = 0
        in_str = False
        esc = False
        for ch in text[start:]:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


def load_qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    root = Path("/home/hobbs/code/lyg/m4/EmotionManifold")
    sys.path.insert(0, str(root))
    from config import ModelConfig

    cfg = ModelConfig("qwen")
    path = str(cfg.local_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.float32, local_files_only=True
    ).to("cpu")
    model.eval()

    class Crit(StoppingCriteria):
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, input_ids, scores, **kwargs):
            return self.inner(input_ids, scores, **kwargs)

    def run(system: str, user: str) -> tuple[str, float]:
        # The published prompts are a system message plus the transcript.
        # This checkpoint is the Instruct model the other local rows load,
        # so the two roles go in through its chat template.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        inner = _JsonReady(tok, prompt_len, 512)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Crit(inner)]),
            )
        dt = (time.perf_counter() - t0) * 1000.0
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        return text, dt

    # One token so the first trial does not pay for kernel warmup.
    warm = tok("warmup", return_tensors="pt")
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
    return run, path


def one_trial(backend: str, call, segment_prompt, emotion_prompt, tagged: str) -> dict:
    seg_msg = [*segment_prompt, {"role": "user", "content": tagged}]
    emo_msg = [*emotion_prompt, {"role": "user", "content": tagged}]
    if backend == "gpt":
        seg_text, seg_ms = call(seg_msg, 0.1)
        emo_text, emo_ms = call(emo_msg, 0.3)
    else:
        seg_text, seg_ms = call(segment_prompt[0]["content"], tagged)
        emo_text, emo_ms = call(emotion_prompt[0]["content"], tagged)
    return {
        "segment_ms": round(seg_ms, 2),
        "emotion_ms": round(emo_ms, 2),
        "label_ms": round(seg_ms + emo_ms, 2),
        "segment_chars": len(seg_text),
        "emotion_chars": len(emo_text),
        "segment_has_json": "{" in seg_text and "}" in seg_text,
        "emotion_has_json": "{" in emo_text and "}" in emo_text,
        "segment_text": seg_text,
        "emotion_text": emo_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpt", "qwen"], required=True)
    ap.add_argument("--tagged", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=PROMPTS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", type=Path, default=None)
    args = ap.parse_args()

    tagged = json.loads(args.tagged.read_text(encoding="utf-8"))
    segment_prompt, emotion_prompt = load_prompts(args.prompts)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.backend == "gpt":
        key = args.key.read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit("empty key file")

        def call(messages, temperature):
            return gpt_call_retry(key, messages, temperature)

        model_note = "gpt-4o-mini"
    else:
        call, model_note = load_qwen()
        print(f"local model: {model_note}", flush=True)

    rows = []
    for n in range(1, 21):
        tid = f"{n:03d}"
        dest = args.out / f"{tid}.json"
        if dest.is_file():
            row = json.loads(dest.read_text(encoding="utf-8"))
            print(f"{tid} skip {row['label_ms']:.0f} ms", flush=True)
        else:
            t0 = time.perf_counter()
            row = one_trial(args.backend, call, segment_prompt, emotion_prompt, tagged[tid])
            row["trial"] = tid
            row["backend"] = args.backend
            row["model"] = model_note
            dest.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"{tid} seg {row['segment_ms']/1000:.1f}s emo {row['emotion_ms']/1000:.1f}s "
                f"sum {row['label_ms']/1000:.1f}s wall {time.perf_counter()-t0:.1f}s "
                f"json {row['segment_has_json']}/{row['emotion_has_json']}",
                flush=True,
            )
        rows.append(row)

    with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "segment_ms", "emotion_ms", "label_ms",
                                          "segment_has_json", "emotion_has_json"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in w.fieldnames})
    print(f"wrote {args.out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
