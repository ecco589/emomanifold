#!/usr/bin/env python3
"""
Fig. nrc_pole_sweep (NIPS fig2) — IEEE conference (RAL/ieeeconf) matplotlib port.

- 阴影带：六模型 min–max（NIPS 五模型 + Qwen2.5-0.5B）
- 实线：六模型 Spearman ρ 算术均值
- 三栏 V/A/D，色板与 appendix_nrc_readout.tex 一致

Reads:
  RAL/experiment/E1/result/pole_sweep_mean_rho.json
Writes:
  RAL/experiment/E1/output/e1_spearman_pole_sweep.{pdf,png,eps}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
E1_DIR = CODE_DIR.parent
RESULT_DIR = E1_DIR / "result"
OUTPUT_DIR = E1_DIR / "output"

SERIES_PATH = RESULT_DIR / "pole_sweep_mean_rho.json"

POLE_NS = [10, 20, 30, 40, 50, 60, 70, 80]
DIMS = ("valence", "arousal", "dominance")
TITLES = {"valence": "V", "arousal": "A", "dominance": "D"}

COLORS = {
    "valence": "#0072B2",
    "arousal": "#D55E00",
    "dominance": "#009E73",
}
FILL_ALPHA = {
    "valence": 0.1875,
    "arousal": 0.154,
    "dominance": 0.154,
}


def _blend_on_white(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    mix = lambda c: (1.0 - alpha) + alpha * c
    return f"#{int(mix(r) * 255):02x}{int(mix(g) * 255):02x}{int(mix(b) * 255):02x}"


YLIMS = {
    "valence": (0.64, 0.86),
    "arousal": (0.32, 0.86),
    "dominance": (0.36, 0.80),
}

IEEE_TEXTWIDTH_IN = 7.0
FIG_HEIGHT_IN = IEEE_TEXTWIDTH_IN * 0.34
IEEE_FONT_SIZE = 8


def _setup_ieee_rcparams():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": IEEE_FONT_SIZE,
            "axes.labelsize": IEEE_FONT_SIZE,
            "axes.titlesize": IEEE_FONT_SIZE,
            "xtick.labelsize": IEEE_FONT_SIZE,
            "ytick.labelsize": IEEE_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 2.0,
        }
    )


def _series_by_pole(points: list[dict]) -> dict[int, dict]:
    return {int(p["pole_n"]): p for p in points}


def _plot_panel(ax, dim: str, pts: list[dict], *, show_ylabel: bool) -> None:
    data = _series_by_pole(pts)
    color = COLORS[dim]
    xs = [n for n in POLE_NS if n in data]
    lo = [data[n]["min_rho"] for n in xs]
    hi = [data[n]["max_rho"] for n in xs]
    mu = [data[n]["mean_rho"] for n in xs]

    fill_color = _blend_on_white(color, FILL_ALPHA[dim])
    ax.fill_between(xs, lo, hi, color=fill_color, linewidth=0, zorder=1)
    ax.plot(xs, mu, color=color, linewidth=2.5, solid_capstyle="round", zorder=3)

    ax.set_title(TITLES[dim])
    ax.set_xlabel("$N$")
    if show_ylabel:
        ax.set_ylabel("Spearman $\\rho$")
    ax.set_xlim(8, 82)
    ax.set_xticks(POLE_NS)
    ax.set_ylim(*YLIMS[dim])
    ax.grid(True, linestyle="--", linewidth=0.4, color="0.85", zorder=0)
    ax.tick_params(
        direction="in",
        length=3,
        width=0.6,
        labelleft=show_ylabel,
        left=show_ylabel,
    )


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，无法出图", file=sys.stderr)
        return 1

    if not SERIES_PATH.is_file():
        print(f"Missing {SERIES_PATH}; run export_pole_sweep_series.py first.", file=sys.stderr)
        return 1

    payload = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    series = payload["dimensions"]

    _setup_ieee_rcparams()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(IEEE_TEXTWIDTH_IN, FIG_HEIGHT_IN),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.88, wspace=0.10)

    for i, (ax, dim) in enumerate(zip(axes, DIMS)):
        _plot_panel(ax, dim, series.get(dim, []), show_ylabel=(i == 0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "e1_spearman_pole_sweep"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.eps", bbox_inches="tight", pad_inches=0.02, format="eps")
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.png")
    print(f"Wrote {stem}.eps")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
