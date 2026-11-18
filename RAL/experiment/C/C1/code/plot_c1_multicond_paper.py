#!/usr/bin/env python3
"""C1 终稿主图（无锚定全量场，四条件对照版）。

A: 方向命中率，四条件 × V/A，trial 聚类 bootstrap 95% CI + 每 trial 散点
B: 示例轨迹（ours，效价 $z$ 分数，固定滞后对齐）
C: token 级相对话轮级的 R² 增益，四条件 × V/A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from run_c1_event_analysis import ema_1d, swing_events, variance_decomposition  # noqa: E402
from run_c1_multicond_compare import (  # noqa: E402
    SYSTEMS,
    attach_command,
    fit_static_calibration,
    load_face,
)

# Single-column (ieeeconf \columnwidth) sizing: (a) full-width on top,
# (b)+(c) side by side below. Height relaxed from the prior 2.60 short form.
IEEE_COLUMNWIDTH_IN = 3.45
FIG_HEIGHT_IN = 5.05
FIG_HEIGHT_IN_2ROW = 3.20
FS_LABEL = 8
FS_TICK = 8
FS_TITLE = 8
FS_LEGEND = 8
FS_ANNOT = 8
FS_PANEL = 8

SYS_LABEL = {
    "ours": "EmoManifold",
    "mishra_gpt": "Mishra",
    "xpress_gpt": "Xpress",
    "random": "Shuffled",
}
# Fig.2 Okabe–Ito accents (plot_e1): blue / green / orange.
COLOR_BLUE = "#0072B2"
COLOR_GREEN = "#009E73"
COLOR_ORANGE = "#D55E00"
SYS_COLOR = {
    "ours": COLOR_BLUE,
    "mishra_gpt": COLOR_GREEN,
    "xpress_gpt": COLOR_ORANGE,
    "random": "#9E9E9E",  # shuffled = neutral gray
}
COLOR_CHANCE = "#555555"  # dashed reference; keep distinct from orange bars


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.7, labelsize=FS_TICK)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def set_panel_title(ax, letter: str, title: str, y: float = 1.04, gap_pt: float = 3.0) -> None:
    """Draw '(x) Title' with the title hugging the letter, independent of axes width.

    A fixed axes-fraction offset (the old ``title_x``) looks fine on a narrow
    panel but leaves a huge gap on a wide one (same fraction = many more
    points). Instead, render the letter first, measure it, and place the
    title `gap_pt` points after its right edge.
    """
    letter_txt = ax.text(0.0, y, letter, transform=ax.transAxes, fontsize=FS_PANEL,
                          fontweight="bold", ha="left", va="bottom", clip_on=False)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = letter_txt.get_window_extent(renderer=renderer)
    x_disp = bbox.x1 + gap_pt * fig.dpi / 72.0
    x_axes, _ = ax.transAxes.inverted().transform((x_disp, 0.0))
    ax.text(x_axes, y, title, transform=ax.transAxes, fontsize=FS_TITLE,
            ha="left", va="bottom", clip_on=False)


def load_all(root: Path, smooth_tau: float, fixed_lag: float) -> dict[str, list[dict]]:
    calib = fit_static_calibration(root)
    cmds = {}
    for tid_dir in sorted((root / "ours").iterdir()):
        if tid_dir.is_dir() and (tid_dir / "source_va.csv").is_file():
            src = pd.read_csv(tid_dir / "source_va.csv")
            if len(src) >= 5:
                cmds[tid_dir.name] = src
    out: dict[str, list[dict]] = {}
    for sysn in SYSTEMS:
        trials = []
        for tid, src in cmds.items():
            face = load_face(root / sysn / tid, smooth_tau, calib)
            if face is None:
                continue
            tr = attach_command(face, src, smooth_tau, clip_command=True)
            tr["lag"] = fixed_lag
            tr["DV"] = np.interp(tr["st"] + fixed_lag, tr["dt"], tr["dv"], left=np.nan, right=np.nan)
            tr["DA"] = np.interp(tr["st"] + fixed_lag, tr["dt"], tr["da"], left=np.nan, right=np.nan)
            trials.append(tr)
        out[sysn] = trials
    return out


def dir_stats(trials: list[dict], dim: str, thr: float, window: float, rng) -> dict:
    df = swing_events(trials, dim, window)
    big = df[df["d_src"].abs() > thr].copy()
    big["hit"] = (np.sign(big["d_src"]) == np.sign(big["d_det"])).astype(float)
    per = big.groupby("id")["hit"].mean().to_numpy()
    boots = np.array([rng.choice(per, size=len(per), replace=True).mean() for _ in range(5000)])
    return dict(
        acc=float(per.mean()),
        lo=float(np.percentile(boots, 2.5)),
        hi=float(np.percentile(boots, 97.5)),
        per_trial=per,
        n_events=int(len(big)),
    )


def panel_a(ax, stats: dict) -> None:
    """stats[(sys, dim)] -> dir_stats dict"""
    from matplotlib.patches import Patch

    dims = [("valence", "Valence"), ("arousal", "Arousal")]
    xs = np.arange(len(dims), dtype=float) * 1.35
    w = 0.22
    rng = np.random.default_rng(1)

    for j, sysn in enumerate(SYSTEMS):
        offset = (j - 1.5) * (w + 0.025)
        for i, (dim, _) in enumerate(dims):
            s = stats[(sysn, dim)]
            x = xs[i] + offset
            acc = s["acc"] * 100
            ax.bar(
                x, acc, width=w, color=SYS_COLOR[sysn], edgecolor="none", zorder=2,
                yerr=[[acc - s["lo"] * 100], [s["hi"] * 100 - acc]],
                error_kw=dict(ecolor="#222222", capsize=2, lw=0.8, capthick=0.8),
            )
            jit = rng.uniform(-w * 0.28, w * 0.28, size=len(s["per_trial"]))
            ax.scatter(
                x + jit, s["per_trial"] * 100, s=6, facecolor="white",
                edgecolor="#333333", linewidth=0.45, zorder=3, alpha=0.85,
            )
            if sysn == "ours":
                y_top = max(float(np.max(s["per_trial"]) * 100), s["hi"] * 100)
                y_ann = y_top + 7.0
                ax.text(x, y_ann, f"{acc:.0f}%", ha="center", va="bottom",
                        fontsize=FS_ANNOT, fontweight="semibold", color="#1a1a1a",
                        clip_on=False, zorder=6)

    ax.axhline(50, color=COLOR_CHANCE, ls="--", lw=1.1, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in dims], fontsize=FS_TICK)
    ax.set_ylabel("Direction agree. (%)", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)

    handles = [Patch(facecolor=SYS_COLOR[s], edgecolor="none", label=SYS_LABEL[s]) for s in SYSTEMS]
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=COLOR_CHANCE, ls="--", lw=1.1, label="Chance"))
    ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=5, borderaxespad=0.0,
              handlelength=0.85, columnspacing=0.28, handletextpad=0.22)
    style_axes(ax)
    set_panel_title(ax, "(a)", "Intra-utterance transitions", y=1.30)


def panel_b(ax, trials_ours: list[dict], trial_id: str) -> dict:
    tr = next(t for t in trials_ours if t["id"] == trial_id)
    m = np.isfinite(tr["DV"])
    t = tr["st"][m] - tr["st"][m][0]

    src_c, face_c = COLOR_BLUE, COLOR_ORANGE
    ax.plot(t, zscore(tr["sv"][m]), color=src_c, lw=1.5, label="Command", zorder=3)
    ax.plot(t, zscore(tr["DV"][m]), color=face_c, lw=1.5, label="Camera", zorder=3)
    ax.axhline(0, color="#D0D0D0", lw=0.7, zorder=1)
    ax.set_ylabel(r"Valence ($z$)", fontsize=FS_LABEL, labelpad=2)
    # Put the unit label in the gap to the right of the axis (level with the
    # tick numbers) instead of on its own centered line below the axis. Kept
    # a size smaller than the main labels since it's a secondary annotation.
    ax.text(1.04, -0.045, "Time (s)", transform=ax.transAxes, fontsize=FS_LABEL - 1.5,
            ha="left", va="center", clip_on=False)
    style_axes(ax)
    # Panel is now narrower (shares a row with panel c); crop to the back half of
    # the trial instead of squeezing the full 0-t_max span into half the width.
    t_max = float(t.max())
    ax.set_xlim(min(3.0, max(t_max - 1.0, 0.0)), t_max)
    ax.set_ylim(-3.5, 4.1)
    ax.set_yticks([-3, 0, 3])
    ax.legend(frameon=False, fontsize=FS_LEGEND - 1.5, loc="upper left",
              bbox_to_anchor=(0.08, 1.06), handlelength=1.2, labelspacing=0.25,
              borderaxespad=0.0)
    cat = tr.get("cat") or ""
    set_panel_title(ax, "(b)", "Trial" + (f" ({cat})" if cat else ""), y=1.08)
    return tr


def panel_c(ax, gains: dict) -> None:
    """gains[(sys, dim)] -> R2 gain (fraction)"""
    dims = [("valence", "Valence"), ("arousal", "Arousal")]
    xs = np.arange(len(dims), dtype=float) * 1.35
    w = 0.22

    for j, sysn in enumerate(SYSTEMS):
        offset = (j - 1.5) * (w + 0.025)
        for i, (dim, _) in enumerate(dims):
            g = gains[(sysn, dim)] * 100
            x = xs[i] + offset
            ax.bar(x, g, width=w, color=SYS_COLOR[sysn], edgecolor="none", zorder=2)
            if sysn == "ours":
                ax.text(x, g + 0.8, f"+{g:.0f} pp", ha="center", va="bottom",
                        fontsize=FS_ANNOT, fontweight="semibold", color="#1a1a1a")

    ax.axhline(0, color="#888888", lw=0.7, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in dims], fontsize=FS_TICK)
    ax.set_ylabel(r"$R^2$ gain (pp)", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylim(-2, 27)
    # Extra left padding so the "+NN pp" annotation above the leftmost bar
    # does not collide with the y-axis spine now that the panel is narrower.
    ax.set_xlim(xs[0] - 0.85, xs[-1] + 0.6)
    style_axes(ax)
    set_panel_title(ax, "(c)", "Beyond turn-level", y=1.06)


def write_caption(out_dir: Path, stats: dict, gains: dict, trial_id: str, n_trials: int, fig_relpath: str) -> None:
    sv = stats[("ours", "valence")]
    sa = stats[("ours", "arousal")]
    md = f"""# Figure caption (C1, four-condition)

**Closed-loop execution of token-level affect under a four-condition, same-session comparison (n = {n_trials} utterances).**
**(a)** Direction agreement between commanded affect transitions (2 s window) and camera readback. Ours: valence {sv['acc']*100:.0f}% (95% CI [{sv['lo']*100:.0f}, {sv['hi']*100:.0f}]), arousal {sa['acc']*100:.0f}% (CI [{sa['lo']*100:.0f}, {sa['hi']*100:.0f}]); all baselines at chance. Bars: trial-mean accuracy with trial-clustered bootstrap 95% CI; open circles: per-trial accuracy (one per utterance); dashed line: chance (50%).
**(b)** Representative trial {trial_id}: fixed-lag-aligned, z-scored commanded vs executed trajectories.
**(c)** Facial variance explained beyond a turn-level constant expression. Token-level control adds +{gains[('ours','valence')]*100:.0f} / +{gains[('ours','arousal')]*100:.0f} pp (valence/arousal); turn-level and shuffled baselines add none by construction.
All conditions share one fixed execution+detection lag (0.5 s); FER readback linearly calibrated on the static-face conditions (calibration set independent of the tested dynamics).
"""
    tex = (
        r"\begin{figure}[t]" "\n"
        r"  \centering" "\n"
        rf"  \includegraphics[width=\columnwidth]{{{fig_relpath}}}" "\n"
        rf"  \caption{{Closed-loop execution of token-level affect under a four-condition, same-session comparison ($n{{=}}{n_trials}$ utterances)." "\n"
        rf"  \textbf{{(a)}}~Direction agreement between commanded affect transitions (2\,s window) and camera readback: ours reaches {sv['acc']*100:.0f}\% (valence, 95\% CI [{sv['lo']*100:.0f}, {sv['hi']*100:.0f}]) and {sa['acc']*100:.0f}\% (arousal), while turn-level (Mishra, Xpress) and time-shuffled baselines stay at chance (dashed line)." "\n"
        r"  Bars: trial-mean accuracy with trial-clustered bootstrap 95\% CIs; open circles: per-trial accuracy (one per utterance)." "\n"
        rf"  \textbf{{(b)}}~Representative trial {trial_id}: fixed-lag-aligned, $z$-scored commanded (LLM) versus executed (camera) trajectories." "\n"
        rf"  \textbf{{(c)}}~Facial variance explained beyond a turn-level constant expression: token-level control adds +{gains[('ours','valence')]*100:.0f}/+{gains[('ours','arousal')]*100:.0f}\,pp (valence/arousal); baselines add none by construction." "\n"
        r"  All conditions share one fixed execution+detection lag (0.5\,s); FER readback is linearly calibrated on the static-face conditions.}" "\n"
        r"  \label{fig:c1_main}" "\n"
        r"\end{figure}" "\n"
    )
    (out_dir / "c1_main_caption.md").write_text(md, encoding="utf-8")
    (out_dir / "c1_main_caption.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=C1_DIR / "result" / "multicond_noanchor_20260731")
    parser.add_argument("--out-dir", type=Path, default=C1_DIR / "output" / "multicond_noanchor_20260731" / "paper")
    parser.add_argument("--example-trial", default="005")
    parser.add_argument("--fixed-lag", type=float, default=0.5)
    parser.add_argument("--smooth-tau", type=float, default=0.6)
    parser.add_argument("--window", type=float, default=2.0)
    args = parser.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    all_trials = load_all(args.root, args.smooth_tau, args.fixed_lag)
    rng = np.random.default_rng(0)
    stats, gains = {}, {}
    for sysn in SYSTEMS:
        trials = all_trials[sysn]
        for dim, thr in (("valence", 0.3), ("arousal", 0.2)):
            stats[(sysn, dim)] = dir_stats(trials, dim, thr, args.window, rng)
            gains[(sysn, dim)] = variance_decomposition(trials, dim)["R2_gain"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(IEEE_COLUMNWIDTH_IN, FIG_HEIGHT_IN_2ROW), dpi=220)
    outer = GridSpec(
        2, 1, figure=fig, height_ratios=[0.95, 1.0],
        hspace=0.42, left=0.155, right=0.99, top=0.88, bottom=0.12,
    )
    ax_a = fig.add_subplot(outer[0])
    inner_bc = outer[1].subgridspec(1, 2, wspace=0.55)
    ax_b = fig.add_subplot(inner_bc[0])
    ax_c = fig.add_subplot(inner_bc[1])

    panel_a(ax_a, stats)
    panel_b(ax_b, all_trials["ours"], args.example_trial)
    panel_c(ax_c, gains)

    pdf = args.out_dir / "c1_main_figure.pdf"
    png = args.out_dir / "c1_main_figure.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    n_trials = len(all_trials["ours"])
    relpath = "experiment/C/C1/output/multicond_noanchor_20260731/paper/c1_main_figure.pdf"
    write_caption(args.out_dir, stats, gains, args.example_trial, n_trials, relpath)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
