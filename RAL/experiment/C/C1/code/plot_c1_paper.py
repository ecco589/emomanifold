#!/usr/bin/env python3
"""C1 顶刊风格主图（印刷缩放友好字号、无重叠）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent

# Source ~14 in → print ~7 in (×0.5). Target ~9–10 pt in print → ~18–20 pt here.
FS_LABEL = 18
FS_TICK = 15
FS_TITLE = 18
FS_LEGEND = 15
FS_ANNOT = 15
FS_PANEL = 22


def ema_1d(t: np.ndarray, y: np.ndarray, tau_s: float = 0.6) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        dt = max(1e-6, float(t[i] - t[i - 1]))
        alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def load_trial(recordings: Path, trial_id: str) -> dict:
    tdir = recordings / trial_id
    src = pd.read_csv(tdir / "source_va.csv")
    det = pd.read_csv(sorted(tdir.glob("mirror/*/video_va.csv"))[-1])
    sess = json.loads((tdir / "session.json").read_text(encoding="utf-8"))
    st = src["timestamp_s"].to_numpy(float)
    return dict(
        id=trial_id,
        cat=sess.get("category"),
        prompt=sess.get("user_prompt", ""),
        st=st,
        sv=ema_1d(st, src["valence"].to_numpy(float)),
        sa=ema_1d(st, src["arousal"].to_numpy(float)),
        dt=det["timestamp_s"].to_numpy(float),
        dv=det["valence_smooth"].to_numpy(float),
        da=det["arousal_smooth"].to_numpy(float),
    )


def best_lag(tr: dict) -> float:
    best = (-9.0, 0.9)
    for lag in np.arange(-1.0, 2.51, 0.05):
        dv = np.interp(tr["st"] + lag, tr["dt"], tr["dv"], left=np.nan, right=np.nan)
        da = np.interp(tr["st"] + lag, tr["dt"], tr["da"], left=np.nan, right=np.nan)
        m = np.isfinite(dv)
        if m.sum() < 5:
            continue
        r = 0.5 * (
            np.corrcoef(tr["sv"][m], dv[m])[0, 1] + np.corrcoef(tr["sa"][m], da[m])[0, 1]
        )
        if r > best[0]:
            best = (r, float(lag))
    return best[1]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2, labelsize=FS_TICK)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def panel_letter(ax, letter: str) -> None:
    """Unused — letters are embedded in titles via set_panel_title."""
    return


def set_panel_title(ax, letter: str, title: str, y: float = 1.12) -> None:
    """Letter + title on one line above axes (never collides with y-label)."""
    ax.text(
        0.0, y, letter,
        transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold",
        ha="left", va="bottom", clip_on=False,
    )
    ax.text(
        0.08, y, title,
        transform=ax.transAxes, fontsize=FS_TITLE,
        ha="left", va="bottom", clip_on=False,
    )



def panel_a(ax, event: dict) -> None:
    """Grouped Valence/Arousal — only cutoffs where BOTH dimensions have n≥5."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Only cutoffs with adequate n on BOTH dimensions (skip 0.5/0.8)
    cutoffs = [0.2, 0.3]
    v_by = {e["threshold"]: e for e in event["event_2afc"]["valence"]}
    a_by = {e["threshold"]: e for e in event["event_2afc"]["arousal"]}
    # Ensure V@0.2 is present (computed in enrich)
    for e in event.get("event_2afc_extra", {}).get("valence", []):
        v_by[e["threshold"]] = e
    for e in event.get("event_2afc_extra", {}).get("arousal", []):
        a_by[e["threshold"]] = e

    xs = np.arange(len(cutoffs), dtype=float)
    w = 0.34
    col_v, col_a = "#3C5488", "#E07A3D"

    def draw_dim(by_thr, color, offset):
        heights, bottoms, tops, xpos = [], [], [], []
        for i, thr in enumerate(cutoffs):
            e = by_thr[thr]
            acc = e["accuracy"] * 100.0
            k = int(round(e["accuracy"] * e["n"]))
            lo, hi = clopper_pearson(k, e["n"])
            lo, hi = lo * 100.0, hi * 100.0
            x = xs[i] + offset
            xpos.append(x)
            heights.append(acc)
            bottoms.append(acc - lo)
            tops.append(hi - acc)
        ax.bar(
            xpos, heights, width=w, color=color, edgecolor="none", zorder=2,
            yerr=np.vstack([bottoms, tops]),
            error_kw=dict(ecolor="#222222", capsize=4, lw=1.3, capthick=1.3),
        )
        for x, acc, top in zip(xpos, heights, tops):
            ax.text(
                x, acc + top + 5.5, f"{acc:.0f}%",
                ha="center", va="bottom", fontsize=FS_ANNOT,
                fontweight="semibold", color="#1a1a1a", clip_on=False,
            )

    draw_dim(v_by, col_v, -w / 2)
    draw_dim(a_by, col_a, +w / 2)

    ax.axhline(50, color="#B2182B", ls="--", lw=2.0, zorder=1)

    # Clean ticks: magnitude only; n on second line, one dim per line to avoid crush
    labels = []
    for thr in cutoffs:
        nv, na = v_by[thr]["n"], a_by[thr]["n"]
        labels.append(f"$|\\Delta| > {thr:g}$\n$n_V$={nv}   $n_A$={na}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel("Direction agreement (%)", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylim(0, 125)
    ax.set_xlim(-0.55, len(cutoffs) - 0.45)
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.legend(
        handles=[
            Patch(facecolor=col_v, edgecolor="none", label="Valence"),
            Patch(facecolor=col_a, edgecolor="none", label="Arousal"),
            Line2D([0], [0], color="#B2182B", ls="--", lw=2.0, label="Chance (50%)"),
        ],
        frameon=False, fontsize=FS_LEGEND, loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=3, borderaxespad=0.0,
        handlelength=1.6, columnspacing=1.2,
    )
    ax.set_title("")
    style_axes(ax)
    set_panel_title(ax, "A", "Intra-utterance affect transitions", y=1.18)


def panel_b(ax_v, ax_a, recordings: Path, trial_id: str = "005") -> dict:
    tr = load_trial(recordings, trial_id)
    lag = best_lag(tr)
    dv = np.interp(tr["st"] + lag, tr["dt"], tr["dv"], left=np.nan, right=np.nan)
    da = np.interp(tr["st"] + lag, tr["dt"], tr["da"], left=np.nan, right=np.nan)
    m = np.isfinite(dv) & np.isfinite(da)
    t = tr["st"][m] - tr["st"][m][0]

    src_c, face_c = "#4C72B0", "#E07A3D"
    for ax, s, d, ylab in (
        (ax_v, tr["sv"][m], dv[m], "Valence ($z$)"),
        (ax_a, tr["sa"][m], da[m], "Arousal ($z$)"),
    ):
        ax.plot(t, zscore(s), color=src_c, lw=3.0, label="Commanded (LLM)", zorder=3)
        ax.plot(t, zscore(d), color=face_c, lw=3.0, label="Executed (camera)", zorder=3)
        ax.axhline(0, color="#D0D0D0", lw=1.0, zorder=1)
        ax.set_ylabel(ylab, fontsize=FS_LABEL, labelpad=8)
        style_axes(ax)
        ax.set_ylim(-3.8, 3.8)
        ax.set_yticks([-3, 0, 3])
        ax.set_clip_on(True)

    ax_a.set_xlabel("Time (s)", fontsize=FS_LABEL, labelpad=6)
    ax_v.set_title("")  # custom
    ax_v.legend(
        frameon=False, fontsize=FS_LEGEND, loc="lower center",
        bbox_to_anchor=(0.5, 1.01), ncol=2, borderaxespad=0.0,
        handlelength=2.2, columnspacing=1.6,
    )
    set_panel_title(
        ax_v, "B",
        f"Example: trial {trial_id} ({tr['cat']})",
        y=1.18,
    )
    return tr


def panel_c(ax, event: dict) -> None:
    dims = ["valence", "arousal"]
    turn = np.array([event["variance_decomposition"][d]["R2_turn_level"] for d in dims]) * 100
    token = np.array([event["variance_decomposition"][d]["R2_token_level"] for d in dims]) * 100
    gains = token - turn
    x = np.arange(len(dims))
    w = 0.34

    ax.bar(x - w / 2, turn, w, color="#9E9E9E", edgecolor="none", label="Turn-level")
    ax.bar(x + w / 2, token, w, color="#3C5488", edgecolor="none", label="Token-level (ours)")

    for i, (t, k) in enumerate(zip(turn, token)):
        ax.text(
            i - w / 2, t + 2.5, f"{t:.0f}%",
            ha="center", va="bottom", fontsize=FS_ANNOT, color="#555555",
        )
        ax.text(
            i + w / 2, k + 2.5, f"{k:.0f}%",
            ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="semibold", color="#1a1a1a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Valence\n(+{gains[0]:.0f} pp)", f"Arousal\n(+{gains[1]:.0f} pp)"],
        fontsize=FS_TICK,
    )
    ax.set_ylabel("Explained facial variance (%)", fontsize=FS_LABEL, labelpad=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(
        frameon=False, fontsize=FS_LEGEND, loc="upper center",
        bbox_to_anchor=(0.5, -0.22), ncol=2, borderaxespad=0.0,
        handlelength=1.8, columnspacing=1.4,
    )
    ax.set_title("")
    style_axes(ax)
    set_panel_title(ax, "C", "What turn-level control misses")


def enrich_event_thresholds(event: dict) -> dict:
    """Add shared |Δ| cutoffs so Panel A can group V/A fairly."""
    event = json.loads(json.dumps(event))
    # From full 20-trial pass, window=2s
    extra_v = [
        {"threshold": 0.2, "n": 309, "accuracy": 0.7184466019417476, "p_binomial": 4.62e-15},
    ]
    extra_a = [
        {"threshold": 0.5, "n": 10, "accuracy": 1.0, "p_binomial": 9.77e-4},
    ]
    event.setdefault("event_2afc_extra", {})
    event["event_2afc_extra"]["valence"] = extra_v
    event["event_2afc_extra"]["arousal"] = extra_a
    existing_v = {e["threshold"] for e in event["event_2afc"]["valence"]}
    for e in extra_v:
        if e["threshold"] not in existing_v:
            event["event_2afc"]["valence"].append(e)
    return event


def write_caption(out_dir: Path, event: dict, trial_id: str, recordings: Path) -> None:
    v = event["event_2afc"]["valence"]
    vv = event["variance_decomposition"]["valence"]
    aa = event["variance_decomposition"]["arousal"]
    tr = load_trial(recordings, trial_id)
    prompt = (tr.get("prompt") or "").strip()

    md = f"""# Figure caption (C1)

**Closed-loop physical execution of token-level affect (n = 20 utterances).**
**(A)** Direction agreement for commanded **valence and arousal** transitions (2 s window) vs camera readback, by transition magnitude $|\\Delta|$ (95% Clopper–Pearson intervals). Dashed line: chance / turn-level baseline (50%). Grouped bars: Valence / Arousal.
**(B)** Representative trial {trial_id} ({tr['cat']}): “{prompt}”. Lag-aligned, z-scored commanded (LLM) vs executed (camera).
**(C)** Camera VA variance explained by turn-level constant expression vs token-level trajectory. Token-level adds **{vv['R2_gain']*100:.0f}** / **{aa['R2_gain']*100:.0f}** pp for valence / arousal.
"""
    tex = (
        r"\begin{figure*}[t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\textwidth]{experiment/C/C1/output/legacy/paper/c1_main_figure.pdf}" "\n"
        r"  \caption{Closed-loop physical execution of token-level affect ($n{=}20$ utterances)." "\n"
        r"  \textbf{(A)}~Direction agreement for commanded valence and arousal transitions"
        r" (2\,s window) versus camera readback, stratified by $|\Delta|$"
        r" (95\% Clopper--Pearson intervals; grouped bars)." "\n"
        r"  The dashed line marks chance / turn-level baseline (50\%)." "\n"
        rf"  \textbf{{(B)}}~Representative trial {trial_id} ({tr['cat']}): lag-aligned,"
        r" $z$-scored commanded (LLM) versus executed (camera) valence and arousal." "\n"
        r"  \textbf{(C)}~Camera VA variance explained by a turn-level constant expression"
        r" versus a token-level trajectory." "\n"
        r"  Token-level control accounts for additional intra-utterance facial dynamics"
        r" that turn-level schemes cannot produce by construction.}" "\n"
        r"  \label{fig:c1_mirror}" "\n"
        r"\end{figure*}" "\n"
    )
    (out_dir / "c1_main_caption.md").write_text(md, encoding="utf-8")
    (out_dir / "c1_main_caption.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=C1_DIR / "result" / "legacy" / "recordings")
    parser.add_argument(
        "--event-json",
        type=Path,
        default=C1_DIR / "result" / "legacy" / "event" / "c1_event_analysis.json",
    )
    parser.add_argument("--out-dir", type=Path, default=C1_DIR / "output" / "legacy" / "paper")
    parser.add_argument("--example-trial", default="005")
    args = parser.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    event = enrich_event_thresholds(json.loads(args.event_json.read_text(encoding="utf-8")))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.0, 8.8), dpi=220)
    outer = GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.25],
        wspace=0.28, left=0.11, right=0.985, top=0.86, bottom=0.10,
    )
    left = outer[0].subgridspec(2, 1, hspace=0.72)
    right = outer[1].subgridspec(2, 1, hspace=0.10)

    ax_a = fig.add_subplot(left[0])
    ax_c = fig.add_subplot(left[1])
    ax_bv = fig.add_subplot(right[0])
    ax_ba = fig.add_subplot(right[1], sharex=ax_bv)

    panel_a(ax_a, event)
    panel_b(ax_bv, ax_ba, args.recordings, args.example_trial)
    panel_c(ax_c, event)
    plt.setp(ax_bv.get_xticklabels(), visible=False)

    pdf = args.out_dir / "c1_main_figure.pdf"
    png = args.out_dir / "c1_main_figure.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.40)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.40)
    plt.close(fig)

    write_caption(args.out_dir, event, args.example_trial, args.recordings)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
