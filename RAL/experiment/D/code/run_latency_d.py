"""
Step D: latency table — text-ready → facial control weights done.

Five conditions × 20 stimuli × --repeats (default 3).
Outputs:
  RAL/experiment/D/out/latency_raw.csv
  RAL/experiment/D/out/latency_table.csv
  RAL/experiment/D/out/machine_env.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES = ROOT / "RAL/experiment/C/C1/code/baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))
B1_CODE = ROOT / "RAL/experiment/B/B1/code"
if str(B1_CODE) not in sys.path:
    sys.path.insert(0, str(B1_CODE))

STIMULI = ROOT / "RAL/experiment/C/C1/data/stimuli.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "out"

FKCN_ORDER = [
    "neutral", "happiness", "surprise", "fear", "sadness", "disgust", "anger",
]


def one_hot(name: str, intensity: float = 1.0) -> list[float]:
    w = [0.0] * 7
    idx = FKCN_ORDER.index(name)
    w[idx] = intensity
    if intensity < 1.0 - 1e-9 and name != "neutral":
        w[0] = 1.0 - intensity
    return w


def load_stimuli(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_env(path: Path, device: str) -> None:
    lines = [
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
    ]
    try:
        import torch
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"mps_available: {torch.backends.mps.is_available()}")
    except Exception as e:
        lines.append(f"torch: unavailable ({e})")
    try:
        brand = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
        if brand:
            lines.append(f"cpu: {brand}")
    except Exception:
        pass
    lines.append(f"device_arg: {device}")
    lines.append(f"proxy: ALL_PROXY={os.environ.get('ALL_PROXY', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument(
        "--conditions",
        default="ours,mishra_gpt,mishra_qwen,xpress_gpt,xpress_qwen",
        help="Comma-separated subset to run",
    )
    ap.add_argument("--stimuli", type=Path, default=STIMULI)
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    need_gpt = any(c.endswith("_gpt") for c in conditions)
    need_qwen = any(c in ("ours", "mishra_qwen", "xpress_qwen") for c in conditions)
    if need_gpt and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY（GPT 条件需要）")

    stimuli = load_stimuli(args.stimuli)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_env(OUT_DIR / "machine_env.txt", args.device)

    # Lazy-loaded shared state
    state: dict = {}

    def ensure_qwen():
        if "loader" in state:
            return
        from src.model_loader import CausalModelLoader
        from src.fkcn import FKCN
        from b1_core import load_e1_projector

        print(f"Loading Qwen on {args.device}...", flush=True)
        loader = CausalModelLoader("qwen", device=args.device)
        loader.load()
        projector, _ = load_e1_projector(loader)
        state["loader"] = loader
        state["projector"] = projector
        state["fkcn"] = FKCN(device="cpu")

    def time_ours(text: str) -> float:
        from src.grimace_utils import smooth_weights_batch

        ensure_qwen()
        loader, projector, fkcn = state["loader"], state["projector"], state["fkcn"]
        t0 = time.perf_counter()
        coords, _tokens = projector.project_text(loader, text, apply_scaling=True)
        points = list(zip(coords["valence"], coords["arousal"]))
        raw = fkcn.compute_weights_batch(points)
        _ = smooth_weights_batch(raw, alpha=0.6)
        return (time.perf_counter() - t0) * 1000

    def time_mishra_gpt(text: str) -> float:
        import run_mishra as m

        prefix = (BASELINES / "mishra_prompt.txt").read_text(encoding="utf-8")
        prompt = m.build_prompt(prefix, text)
        t0 = time.perf_counter()
        raw, _ = m.run_gpt(prompt, args.gpt_model, base_url=None)
        eid = m.parse_emotion_id(raw)
        if eid is None:
            raise RuntimeError(f"mishra gpt parse failed: {raw!r}")
        _ = one_hot(m.ID_TO_NAME[eid], 1.0)
        return (time.perf_counter() - t0) * 1000

    def time_mishra_qwen(text: str) -> float:
        import run_mishra as m

        ensure_qwen()
        # Bind loader into run_mishra.run_qwen cache so it reuses our loaded model
        if not hasattr(m.run_qwen, "_loader"):
            m.run_qwen._loader = state["loader"]
            tok = state["loader"].tokenizer
            cand = {}
            for d in range(1, 7):
                ids = set()
                for s in (str(d), f" {d}", f"{d})", f" {d})"):
                    enc = tok.encode(s, add_special_tokens=False)
                    if enc:
                        ids.add(enc[0])
                cand[d] = sorted(ids)
            m.run_qwen._digit_cands = cand

        prefix = (BASELINES / "mishra_prompt.txt").read_text(encoding="utf-8")
        prompt = m.build_prompt(prefix, text)
        t0 = time.perf_counter()
        raw, _ = m.run_qwen(prompt, args.device)
        eid = m.parse_emotion_id(raw)
        if eid is None:
            raise RuntimeError(f"mishra qwen parse failed: {raw!r}")
        _ = one_hot(m.ID_TO_NAME[eid], 1.0)
        return (time.perf_counter() - t0) * 1000

    def time_xpress_gpt(text: str, sid: str) -> float:
        import run_xpress_conv as x

        system = (BASELINES / "xpress_conv_prompt.txt").read_text(encoding="utf-8").strip()
        label_map = json.loads((BASELINES / "xpress_label_map.json").read_text(encoding="utf-8"))
        row = {"id": sid, "user_prompt": text}
        segs = x.segments_for_row(row)
        t0 = time.perf_counter()
        prev = ""
        cur_expr = "neutral"
        mapped = "neutral"
        inten = "high"
        for seg in segs:
            user = x.build_user_message(prev, seg, cur_expr, "5 seconds")
            raw, _ = x.run_gpt(system, user, args.gpt_model, base_url=None)
            emo, inten = x.parse_reaction(raw)
            if emo is None or inten is None:
                raise RuntimeError(f"xpress gpt parse failed: {raw!r}")
            mapped = x.map_emotion(emo, label_map)
            if emo != "no-change":
                cur_expr = emo
            prev = (prev + " " + seg).strip() if prev else seg
        intensity = 0.5 if inten == "low" else 1.0
        _ = one_hot(mapped, intensity)
        return (time.perf_counter() - t0) * 1000

    def time_xpress_qwen(text: str, sid: str) -> float:
        import run_xpress_conv as x

        ensure_qwen()
        if not hasattr(x.run_qwen, "_loader"):
            x.run_qwen._loader = state["loader"]

        system = (BASELINES / "xpress_conv_prompt.txt").read_text(encoding="utf-8").strip()
        label_map = json.loads((BASELINES / "xpress_label_map.json").read_text(encoding="utf-8"))
        row = {"id": sid, "user_prompt": text}
        segs = x.segments_for_row(row)
        t0 = time.perf_counter()
        prev = ""
        cur_expr = "neutral"
        mapped = "neutral"
        inten = "high"
        for seg in segs:
            user = x.build_user_message(prev, seg, cur_expr, "5 seconds")
            raw, _ = x.run_qwen(system, user, args.device)
            emo, inten = x.parse_reaction(raw)
            if emo is None or inten is None:
                # still count wall time; mark later
                raise RuntimeError(f"xpress qwen parse failed: {raw!r}")
            mapped = x.map_emotion(emo, label_map)
            if emo != "no-change":
                cur_expr = emo
            prev = (prev + " " + seg).strip() if prev else seg
        intensity = 0.5 if inten == "low" else 1.0
        _ = one_hot(mapped, intensity)
        return (time.perf_counter() - t0) * 1000

    timers = {
        "ours": lambda row: time_ours(row["user_prompt"]),
        "mishra_gpt": lambda row: time_mishra_gpt(row["user_prompt"]),
        "mishra_qwen": lambda row: time_mishra_qwen(row["user_prompt"]),
        "xpress_gpt": lambda row: time_xpress_gpt(row["user_prompt"], row["id"]),
        "xpress_qwen": lambda row: time_xpress_qwen(row["user_prompt"], row["id"]),
    }

    raw_path = OUT_DIR / "latency_raw.csv"
    fieldnames = ["condition", "id", "repeat", "latency_ms", "error"]
    raw_rows: list[dict] = []

    # Warmup local paths once
    if need_qwen:
        ensure_qwen()
        _ = time_ours(stimuli[0]["user_prompt"])
        print("Warmup ours done", flush=True)

    for cond in conditions:
        if cond not in timers:
            raise SystemExit(f"Unknown condition: {cond}")
        print(f"=== {cond} ===", flush=True)
        for rep in range(1, args.repeats + 1):
            for row in stimuli:
                err = ""
                ms = float("nan")
                try:
                    ms = timers[cond](row)
                except Exception as e:
                    err = str(e)
                raw_rows.append(
                    {
                        "condition": cond,
                        "id": row["id"],
                        "repeat": rep,
                        "latency_ms": f"{ms:.2f}" if ms == ms else "",
                        "error": err,
                    }
                )
                status = err if err else f"{ms:.1f}ms"
                print(f"[{cond} r{rep} {row['id']}] {status}", flush=True)

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    # Summary
    table_path = OUT_DIR / "latency_table.csv"
    summary = []
    for cond in conditions:
        vals = [
            float(r["latency_ms"])
            for r in raw_rows
            if r["condition"] == cond and r["latency_ms"] and not r["error"]
        ]
        n_err = sum(1 for r in raw_rows if r["condition"] == cond and r["error"])
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            med = statistics.median(vals)
        else:
            mean = std = med = float("nan")
        summary.append(
            {
                "condition": cond,
                "n_ok": len(vals),
                "n_error": n_err,
                "mean_ms": f"{mean:.2f}" if mean == mean else "",
                "std_ms": f"{std:.2f}" if std == std else "",
                "median_ms": f"{med:.2f}" if med == med else "",
                "mean_pm_std": f"{mean:.1f} ± {std:.1f}" if mean == mean else "",
            }
        )
        print(
            f"SUMMARY {cond}: {summary[-1]['mean_pm_std']} "
            f"(n={len(vals)}, err={n_err})",
            flush=True,
        )

    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "n_ok", "n_error", "mean_ms", "std_ms", "median_ms", "mean_pm_std",
            ],
        )
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote {raw_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {OUT_DIR / 'machine_env.txt'}")


if __name__ == "__main__":
    main()
