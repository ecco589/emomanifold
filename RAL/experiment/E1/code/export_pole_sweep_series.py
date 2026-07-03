#!/usr/bin/env python3
"""
六模型 Spearman 汇总（NIPS 五模型 + Qwen2.5-0.5B）→ Fig. nrc_pole_sweep 格式。

Reads:
  RAL/experiment/E1/data/nips_b1_per_model_spearman.json
  RAL/experiment/E1/result/en/pole_<N>/report.json  (qwen25_0_5b)
Writes:
  RAL/experiment/E1/result/pole_sweep_mean_rho.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent
E1_DIR = CODE_DIR.parent
DATA_DIR = E1_DIR / "data"
RESULT_DIR = E1_DIR / "result"
NIPS_PER_MODEL = DATA_DIR / "nips_b1_per_model_spearman.json"
OUT = RESULT_DIR / "pole_sweep_mean_rho.json"

NIPS_MODELS = [
    "gemma22_2b",
    "llama32_3b",
    "mistral7b_v03",
    "qwen25_7b",
    "llama31_8b",
]
QWEN_MODEL = "qwen25_0_5b"
ALL_MODELS = NIPS_MODELS + [QWEN_MODEL]
POLE_NS = [10, 20, 30, 40, 50, 60, 70, 80]
DIMS = ("valence", "arousal", "dominance")


def load_qwen_rho(pole_n: int, dim: str) -> Optional[float]:
    path = RESULT_DIR / "en" / f"pole_{pole_n}" / "report.json"
    if not path.is_file():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    rho = rep.get("spearman", {}).get(dim, {}).get("rho")
    return None if rho is None else float(rho)


def main() -> int:
    if not NIPS_PER_MODEL.is_file():
        print(f"Missing {NIPS_PER_MODEL}", file=sys.stderr)
        return 1

    nips = json.loads(NIPS_PER_MODEL.read_text(encoding="utf-8"))
    by_pole = nips["by_pole"]

    per_model_series = {mk: {d: [] for d in DIMS} for mk in ALL_MODELS}
    agg_series: dict[str, list[dict]] = {d: [] for d in DIMS}

    for pn in POLE_NS:
        nips_cell = by_pole.get(str(pn), {})
        for dim in DIMS:
            vals = []
            for mk in NIPS_MODELS:
                rho = nips_cell.get(mk, {}).get(dim)
                if rho is not None:
                    v = float(rho)
                    vals.append(v)
                    per_model_series[mk][dim].append({"pole_n": pn, "rho": round(v, 6)})
            qrho = load_qwen_rho(pn, dim)
            if qrho is not None:
                vals.append(qrho)
                per_model_series[QWEN_MODEL][dim].append({"pole_n": pn, "rho": round(qrho, 6)})
            if len(vals) != len(ALL_MODELS):
                print(
                    f"warn: pole {pn} {dim}: expected {len(ALL_MODELS)} models, got {len(vals)}",
                    file=sys.stderr,
                )
            if not vals:
                continue
            mu = float(statistics.mean(vals))
            lo = float(min(vals))
            hi = float(max(vals))
            agg_series[dim].append(
                {
                    "pole_n": pn,
                    "mean_rho": round(mu, 4),
                    "min_rho": round(lo, 4),
                    "max_rho": round(hi, 4),
                }
            )

    out = {
        "source": f"{NIPS_PER_MODEL.name} + result/en/pole_<N>/report.json",
        "description": (
            "Six-model Spearman mean and min–max band (NIPS five models + Qwen2.5-0.5B); "
            "NIPS intersection hold-out for Qwen; original B1 hold-out for baseline five."
        ),
        "models": ALL_MODELS,
        "per_model": {
            mk: per_model_series[mk] for mk in ALL_MODELS
        },
        "dimensions": agg_series,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
