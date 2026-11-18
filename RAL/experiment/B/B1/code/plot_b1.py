#!/usr/bin/env python3
"""
Fig. B1 — contextual sarcasm raincloud (S / P / N) + keyword inset.

IEEE/RAL (ieeeconf) matplotlib port; styling aligned with E1 plot_e1.py.

Reads:
  RAL/experiment/B/B1/result/b1_full_metrics.json
  RAL/experiment/B/B1/data/context_triplets_en.json  (primary keywords)
Writes:
  RAL/experiment/B/B1/output/b1_valence_raincloud.{pdf,png,eps}
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
B1_DIR = CODE_DIR.parent
RESULT_PATH = B1_DIR / "result" / "b1_full_metrics.json"
TRIPLETS_PATH = B1_DIR / "data" / "context_triplets_en.json"
OUTPUT_DIR = B1_DIR / "output"

COLORS = {
    "sarcastic": "#0072B2",
    "positive": "#009E73",
    "negative": "#D55E00",
}

# Half-column (single-column) figure for ieeeconf two-column layout.
IEEE_COLUMNWIDTH_IN = 3.45
FIG_HEIGHT_IN = 2.0
IEEE_FONT_SIZE = 8
VIOLIN_WIDTH = 0.24
JITTER_WIDTH = 0.055
N_BOOT = 2000
RNG = np.random.default_rng(42)

# Bracket zone sits above data cloud; reserved in ylim before drawing stats.
BRACKET_STEP_FRAC = 0.048   # per tier; scaled up for 3-pair panels below
BRACKET_BASE_FRAC = 0.05
BRACKET_TOP_INSET = 0.030   # headroom below y_top for the top bracket bar
BRACKET_LABEL_FRAC = 0.022


def _setup_ieee_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": IEEE_FONT_SIZE,
            "axes.labelsize": IEEE_FONT_SIZE,
            "axes.titlesize": IEEE_FONT_SIZE,
            "xtick.labelsize": IEEE_FONT_SIZE - 0.5,
            "ytick.labelsize": IEEE_FONT_SIZE - 0.5,
            "legend.fontsize": IEEE_FONT_SIZE - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.35,
            "lines.linewidth": 0.8,
        }
    )


def _load_triplet_matrix(payload: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_id: Dict[int, Dict[str, float]] = {}
    for row in payload["sentence"]:
        by_id.setdefault(row["id"], {})[row["type"]] = row["sentence_v"]
    ids = sorted(by_id)
    s = np.array([by_id[i]["sarcastic"] for i in ids])
    p = np.array([by_id[i]["positive"] for i in ids])
    n = np.array([by_id[i]["negative"] for i in ids])
    return s, p, n


def _primary_keyword_v(payload: dict, primary_map: dict) -> Tuple[np.ndarray, np.ndarray]:
    kw_by_id = {}
    for row in payload["keywords"]:
        kid = row["id"]
        primary = primary_map.get(kid)
        if primary is None or row["keyword"] != primary:
            continue
        kw_by_id[kid] = row
    ids = sorted(kw_by_id)
    vs = np.array([kw_by_id[i]["valence_sarcastic"] for i in ids], dtype=float)
    vp = np.array([kw_by_id[i]["valence_positive"] for i in ids], dtype=float)
    mask = np.isfinite(vs) & np.isfinite(vp)
    return vs[mask], vp[mask]


def _bootstrap_ci(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    data = np.asarray(data, dtype=float)
    mean = float(np.mean(data))
    boots = [float(np.mean(RNG.choice(data, size=len(data), replace=True))) for _ in range(N_BOOT)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def _data_y_limits(arrays: List[np.ndarray], *, pad_lo: float = 0.06, pad_hi: float = 0.06) -> Tuple[float, float]:
    allv = np.concatenate(arrays)
    ylo = float(np.quantile(allv, 0.01)) - pad_lo
    yhi = float(np.quantile(allv, 0.99)) + pad_hi
    ylo = max(-0.45, ylo)
    return ylo, yhi


def _ylim_with_bracket_zone(
    ylo: float,
    y_data_hi: float,
    n_brackets: int,
    y_span: float,
) -> Tuple[float, float, float]:
    step_frac = {1: 0.070, 2: 0.062, 3: 0.088}.get(n_brackets, BRACKET_STEP_FRAC)
    step = step_frac * y_span
    label_h = BRACKET_LABEL_FRAC * y_span
    bracket_base = y_data_hi + BRACKET_BASE_FRAC * y_span
    zone_top = bracket_base + n_brackets * step + label_h + 0.04 * y_span
    return ylo, zone_top, bracket_base


def _half_violin(ax, data: np.ndarray, x: float, color: str, side: str = "right") -> None:
    from scipy.stats import gaussian_kde

    data = np.asarray(data, dtype=float)
    if len(data) < 2:
        return
    kde = gaussian_kde(data, bw_method="scott")
    y_grid = np.linspace(data.min() - 0.04, data.max() + 0.04, 100)
    dens = kde(y_grid)
    dens = dens / dens.max() * VIOLIN_WIDTH
    if side == "right":
        ax.fill_betweenx(y_grid, x, x + dens, facecolor=color, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(x + dens, y_grid, color=color, alpha=0.50, linewidth=0.55, zorder=2)
    else:
        ax.fill_betweenx(y_grid, x - dens, x, facecolor=color, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(x - dens, y_grid, color=color, alpha=0.50, linewidth=0.55, zorder=2)


def _draw_box_iqr(ax, data: np.ndarray, x: float, color: str) -> None:
    q1, med, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    ax.vlines(x, q1, q3, color=color, linewidth=0.9, zorder=4)
    ax.scatter([x], [med], s=10, color="white", edgecolors=color, linewidths=0.7, zorder=5)


def _jitter_strip(ax, data: np.ndarray, x: float, color: str) -> None:
    jitter = RNG.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(data))
    ax.scatter(x + jitter, data, s=6, c=color, alpha=0.22, linewidths=0, zorder=3)


def _mean_diamond(ax, data: np.ndarray, x: float, color: str) -> None:
    mean, lo, hi = _bootstrap_ci(data)
    ax.errorbar(
        x, mean,
        yerr=[[mean - lo], [hi - mean]],
        fmt="D", markersize=4.2, color=color,
        markeredgecolor="white", markeredgewidth=0.35,
        ecolor=color, elinewidth=0.8, capsize=2.2, zorder=6,
    )


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _bracket_label(a: np.ndarray, b: np.ndarray) -> str:
    # Effect sizes (Delta V) are already reported in the text; keep the
    # in-figure annotation to just the significance stars to cut clutter.
    return _p_to_stars(_wilcoxon_p(a, b))


def _significance_bracket(
    ax, x1: float, x2: float, y: float, text: str, y_span: float, flip: bool = False,
) -> None:
    h = 0.010 * y_span
    label_gap = 0.010 * y_span
    if flip:
        # Bracket sits below its data (data is above `y`): mirror the shape so
        # the bar is the part farthest from the data and the label hugs the
        # data-facing side, instead of literally reusing the "hangs below" bar.
        h, label_gap = -h, -label_gap
    va = "bottom" if flip else "top"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="0.30", linewidth=0.55, clip_on=False, zorder=7)
    ax.text(
        (x1 + x2) / 2, y - label_gap, text,
        ha="center", va=va, fontsize=7.0, clip_on=False, zorder=8,
    )


def _plot_raincloud_panel(
    ax,
    arrays: List[np.ndarray],
    cond_keys: Tuple[str, ...],
    *,
    x_labels: Tuple[str, ...],
    panel_label: str,
    show_ylabel: bool = True,
    ylabel: str = "Sentence $V$",
    ylim_override: Optional[Tuple[float, float]] = None,
    draw_brackets: bool = True,
    bracket_at_bottom: bool = False,
) -> None:
    n = len(arrays)
    x_pos = np.arange(n, dtype=float)
    sides = ("right", "left", "right", "left")

    ylo, y_data_hi = _data_y_limits(arrays)
    y_span_data = y_data_hi - ylo
    n_brackets = len(list(combinations(range(n), 2))) if draw_brackets else 0

    if ylim_override:
        ylo, y_top = ylim_override
        y_span = y_top - ylo
        step_frac = {1: 0.070, 2: 0.062, 3: 0.082}.get(n_brackets, BRACKET_STEP_FRAC)
        step = step_frac * y_span
        h = 0.010 * y_span
        inset = BRACKET_TOP_INSET * y_span
        if bracket_at_bottom:
            # Mirror of the top-zone calc, anchored at ylo instead of y_top:
            # the narrowest pair (k=0 below) sits closest to the data above,
            # the widest pair sits farthest down, near ylo.
            bracket_far_y = ylo + inset + h
            bracket_base = bracket_far_y + max(0, n_brackets - 1) * step
        else:
            bracket_top_y = y_top - inset - h
            bracket_base = bracket_top_y - max(0, n_brackets - 1) * step
    else:
        ylo, y_top, bracket_base = _ylim_with_bracket_zone(ylo, y_data_hi, n_brackets, y_span_data)
        y_span = y_top - ylo

    for i, (arr, key) in enumerate(zip(arrays, cond_keys)):
        color = COLORS[key]
        _half_violin(ax, arr, x_pos[i], color, side=sides[i % len(sides)])
        _draw_box_iqr(ax, arr, x_pos[i], color)
        _jitter_strip(ax, arr, x_pos[i], color)
        _mean_diamond(ax, arr, x_pos[i], color)

    mat = np.column_stack(arrays)
    for row in mat:
        ax.plot(x_pos, row, color="0.72", alpha=0.08, linewidth=0.45, zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(panel_label, loc="left", fontweight="normal", fontsize=IEEE_FONT_SIZE, pad=4)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.30, color="0.90", alpha=0.85, zorder=0)
    ax.tick_params(direction="in", length=2.5, width=0.55)
    ax.set_xlim(-0.50, max(n - 1, 1) + 0.50)
    ax.set_ylim(ylo, y_top)

    if draw_brackets and n_brackets:
        step_frac = {1: 0.070, 2: 0.062, 3: 0.082}.get(n_brackets, BRACKET_STEP_FRAC)
        step = step_frac * y_span
        if bracket_at_bottom:
            step = -step  # k increases -> move away from the data, toward ylo
        pairs = sorted(combinations(range(n), 2), key=lambda ij: (ij[1] - ij[0], ij[0]))
        for k, (i, j) in enumerate(pairs):
            _significance_bracket(
                ax, x_pos[i], x_pos[j],
                bracket_base + k * step,
                _bracket_label(arrays[i], arrays[j]),
                y_span,
                flip=bracket_at_bottom,
            )


def main() -> int:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib 未安装，无法出图", file=sys.stderr)
        return 1

    if not RESULT_PATH.is_file():
        print(f"Missing {RESULT_PATH}; run run_b1_full.py first.", file=sys.stderr)
        return 1

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    triplets_raw = json.loads(TRIPLETS_PATH.read_text(encoding="utf-8"))
    primary_map = {
        item["id"]: item["keywords"][0]
        for item in triplets_raw["items"]
        if item.get("keywords")
    }

    s, p, n_arr = _load_triplet_matrix(payload)
    kw_s, kw_p = _primary_keyword_v(payload, primary_map)

    _setup_ieee_rcparams()

    fig = plt.figure(figsize=(IEEE_COLUMNWIDTH_IN, FIG_HEIGHT_IN))
    gs = GridSpec(
        1, 2, width_ratios=[1.30, 1.35], wspace=0.42,
        left=0.12, right=0.995, bottom=0.19, top=0.90,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_inset = fig.add_subplot(gs[0, 1])

    _plot_raincloud_panel(
        ax_main,
        [s, p, n_arr],
        ("sarcastic", "positive", "negative"),
        x_labels=("S", "P", "N"),
        panel_label="(a) Sentence $V$",
        show_ylabel=True,
        ylabel="$V$",
        ylim_override=(-0.2, 1.0),
    )

    _plot_raincloud_panel(
        ax_inset,
        [kw_s, kw_p],
        ("sarcastic", "positive"),
        x_labels=("S", "P"),
        panel_label="(b) Token $V$",
        show_ylabel=False,
        ylim_override=(-1.0, 1.0),
    )
    ax_inset.tick_params(labelleft=True)
    # Give (b) a touch more horizontal room for tick labels / brackets.
    ax_inset.set_xlim(-0.65, 1.65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "b1_valence_raincloud"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.eps", bbox_inches="tight", pad_inches=0.04, format="eps", transparent=False)
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.png")
    print(f"Wrote {stem}.eps")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
