#!/usr/bin/env python3
"""C1 Mirror 再分析：时间滞后 + 系统偏差 + 执行侧平滑（动态匹配）。

相对原始 compare_metrics（固定 +150 ms、原始尺度 CCC）：
1. 扫描最优时间滞后（机械延迟）
2. 去均值偏差（systematic bias）
3. 对 source 做 EMA 平滑，匹配实机执行/检测侧的低频动态
4. 仿射校准后报告 MAE/CCC（注明为事后尺度对齐，主结论仍用 Pearson）

输出（默认）：
  result/legacy/reanalysis/best_lag/c1_reanalysis_summary.json
  result/legacy/reanalysis/best_lag/c1_reanalysis_trials.csv
  output/legacy/reanalysis/best_lag/c1_reanalysis.md
  output/legacy/reanalysis/best_lag/c1_reanalysis_*.png

固定滞后示例：
  python run_c1_reanalyze.py --fixed-lag 0.8 \\
    --out-dir ../output/legacy/reanalysis/fixed800 \\
    --result-dir ../result/legacy/reanalysis/fixed800
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent
DEFAULT_OUT = C1_DIR / "output" / "legacy" / "reanalysis" / "best_lag"
DEFAULT_RESULT = C1_DIR / "result" / "legacy" / "reanalysis" / "best_lag"


def concordance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def ema_1d(t: np.ndarray, y: np.ndarray, tau_s: float) -> np.ndarray:
    """Causal EMA with time constant tau_s (seconds)."""
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        dt = max(1e-6, float(t[i] - t[i - 1]))
        alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out


def sample_detected(
    src_t: np.ndarray,
    det_t: np.ndarray,
    det_y: np.ndarray,
    lag_s: float,
) -> np.ndarray:
    query = src_t + lag_s
    y = np.interp(query, det_t, det_y, left=np.nan, right=np.nan)
    return y


def affine_map(src: np.ndarray, det: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit det ≈ a * src + b; return a, b, calibrated_det in source scale: (det - b) / a."""
    X = np.vstack([src, np.ones_like(src)]).T
    a, b = np.linalg.lstsq(X, det, rcond=None)[0]
    if abs(a) < 1e-8:
        return float(a), float(b), det.copy()
    calibrated = (det - b) / a
    return float(a), float(b), calibrated


def pack_metrics(src: np.ndarray, det: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(src)),
        "pearson_r": pearson(src, det),
        "ccc": concordance(src, det),
        "mae": mae(src, det),
        "bias": float(np.mean(det - src)),
        "src_std": float(np.std(src)),
        "det_std": float(np.std(det)),
        "std_ratio": float(np.std(det) / (np.std(src) + 1e-12)),
    }


def find_trial_paths(trial_dir: Path) -> Optional[Tuple[Path, Path, dict]]:
    src = trial_dir / "source_va.csv"
    sess_path = trial_dir / "session.json"
    if not src.is_file():
        return None
    mirror_runs = sorted(trial_dir.glob("mirror/*/video_va.csv"))
    if not mirror_runs:
        return None
    # Prefer newest run by parent name timestamp
    video_va = mirror_runs[-1]
    sess = json.loads(sess_path.read_text(encoding="utf-8")) if sess_path.is_file() else {}
    return src, video_va, sess


def analyze_trial(
    trial_id: str,
    src_path: Path,
    video_va_path: Path,
    sess: dict,
    lag_grid: np.ndarray,
    smooth_tau_s: float,
    fixed_lag_s: Optional[float],
) -> Dict[str, Any]:
    src = pd.read_csv(src_path)
    det = pd.read_csv(video_va_path)

    src_t = src["timestamp_s"].to_numpy(dtype=float)
    src_v = src["valence"].to_numpy(dtype=float)
    src_a = src["arousal"].to_numpy(dtype=float)
    det_t = det["timestamp_s"].to_numpy(dtype=float)
    det_v = det["valence_smooth"].to_numpy(dtype=float)
    det_a = det["arousal_smooth"].to_numpy(dtype=float)

    # Smooth source to approximate execution + detector bandwidth
    src_v_s = ema_1d(src_t, src_v, smooth_tau_s)
    src_a_s = ema_1d(src_t, src_a, smooth_tau_s)

    def score_lag(lag: float) -> float:
        dv = sample_detected(src_t, det_t, det_v, lag)
        da = sample_detected(src_t, det_t, det_a, lag)
        m = np.isfinite(dv) & np.isfinite(da)
        if m.sum() < 5:
            return -1e9
        # Score on smoothed source (fairer under execution smoothing)
        return 0.5 * (
            pearson(src_v_s[m], dv[m]) + pearson(src_a_s[m], da[m])
        )

    lag_scores = [(float(lag), score_lag(float(lag))) for lag in lag_grid]
    best_lag, best_score = max(lag_scores, key=lambda x: x[1])
    use_lag = float(fixed_lag_s) if fixed_lag_s is not None else best_lag

    def slice_at(lag: float, use_smooth_src: bool):
        dv = sample_detected(src_t, det_t, det_v, lag)
        da = sample_detected(src_t, det_t, det_a, lag)
        m = np.isfinite(dv) & np.isfinite(da)
        sv = src_v_s[m] if use_smooth_src else src_v[m]
        sa = src_a_s[m] if use_smooth_src else src_a[m]
        return sv, sa, dv[m], da[m]

    layers: Dict[str, Any] = {}

    # L0: original protocol (+150 ms, raw source)
    sv, sa, dv, da = slice_at(0.15, use_smooth_src=False)
    layers["L0_raw_150ms"] = {
        "lag_s": 0.15,
        "valence": pack_metrics(sv, dv),
        "arousal": pack_metrics(sa, da),
    }

    # L1: lag-corrected, raw source (shape only)
    sv, sa, dv, da = slice_at(use_lag, use_smooth_src=False)
    layers["L1_lag"] = {
        "lag_s": use_lag,
        "valence": pack_metrics(sv, dv),
        "arousal": pack_metrics(sa, da),
    }

    # L2: lag + smooth source (match execution bandwidth)
    sv, sa, dv, da = slice_at(use_lag, use_smooth_src=True)
    layers["L2_lag_smooth_src"] = {
        "lag_s": use_lag,
        "smooth_tau_s": smooth_tau_s,
        "valence": pack_metrics(sv, dv),
        "arousal": pack_metrics(sa, da),
    }

    # L3: + debias (remove mean offset)
    def debias(src_y, det_y):
        return det_y - np.mean(det_y - src_y)

    dv_b = debias(sv, dv)
    da_b = debias(sa, da)
    layers["L3_lag_smooth_debias"] = {
        "lag_s": use_lag,
        "smooth_tau_s": smooth_tau_s,
        "valence": pack_metrics(sv, dv_b),
        "arousal": pack_metrics(sa, da_b),
        "bias_removed_V": float(np.mean(dv - sv)),
        "bias_removed_A": float(np.mean(da - sa)),
    }

    # L4: + affine amplitude calibration (detector → source scale)
    aV, bV, dv_c = affine_map(sv, dv)
    aA, bA, da_c = affine_map(sa, da)
    layers["L4_lag_smooth_affine"] = {
        "lag_s": use_lag,
        "smooth_tau_s": smooth_tau_s,
        "valence": pack_metrics(sv, dv_c),
        "arousal": pack_metrics(sa, da_c),
        "affine_V": {"a": aV, "b": bV},
        "affine_A": {"a": aA, "b": aA and bA},
        "note": "Post-hoc scale mapping; primary claim should use Pearson from L2/L3.",
    }
    # fix typo in affine_A b
    layers["L4_lag_smooth_affine"]["affine_A"] = {"a": aA, "b": bA}

    return {
        "trial_id": trial_id,
        "category": sess.get("category"),
        "user_prompt": sess.get("user_prompt"),
        "best_lag_s": best_lag,
        "best_lag_score": best_score,
        "used_lag_s": use_lag,
        "smooth_tau_s": smooth_tau_s,
        "n_tokens": int(len(src)),
        "trial_mean": {
            "src_V": float(src_v.mean()),
            "src_A": float(src_a.mean()),
            "det_V": float(det_v.mean()),
            "det_A": float(det_a.mean()),
        },
        "layers": layers,
        "aligned_for_plot": {
            "t": (src_t + use_lag).tolist(),
            "src_V": src_v_s.tolist(),
            "src_A": src_a_s.tolist(),
            "det_V": sample_detected(src_t, det_t, det_v, use_lag).tolist(),
            "det_A": sample_detected(src_t, det_t, det_a, use_lag).tolist(),
            "det_V_cal": None,  # filled below
            "det_A_cal": None,
        },
    }


def mean_layer(trials: List[dict], layer: str, dim: str, key: str) -> float:
    vals = []
    for t in trials:
        v = t["layers"][layer][dim].get(key)
        if v is not None and v == v:  # not NaN
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def write_plots(trials: List[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Layer comparison bars
    layers = [
        ("L0_raw_150ms", "L0 raw\n+150ms"),
        ("L1_lag", "L1 lag"),
        ("L2_lag_smooth_src", "L2 +smooth\nsrc"),
        ("L3_lag_smooth_debias", "L3 +debias"),
        ("L4_lag_smooth_affine", "L4 +affine"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for ax, dim, title in zip(axes, ("valence", "arousal"), ("Valence", "Arousal")):
        xs = np.arange(len(layers))
        r = [mean_layer(trials, lid, dim, "pearson_r") for lid, _ in layers]
        c = [mean_layer(trials, lid, dim, "ccc") for lid, _ in layers]
        w = 0.35
        ax.bar(xs - w / 2, r, w, label="Pearson r", color="#4C72B0")
        ax.bar(xs + w / 2, c, w, label="CCC", color="#DD8452")
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in layers], fontsize=8)
        ax.set_ylim(-0.2, 1.05)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("C1 reanalysis: lag + smooth + bias / scale corrections")
    fig.tight_layout()
    fig.savefig(out_dir / "c1_reanalysis_layers.png")
    plt.close(fig)

    # 2) Example curves: first 3 trials, L3 view (smoothed src vs debiased det, then also calibrated)
    n_show = min(3, len(trials))
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.8 * n_show), dpi=140, sharex=False)
    if n_show == 1:
        axes = np.array([axes])
    for i in range(n_show):
        t = trials[i]
        lag = t["used_lag_s"]
        # reload paths from trial_id
        trial_dir = C1_DIR / "result" / "legacy" / "recordings" / t["trial_id"]
        paths = find_trial_paths(trial_dir)
        assert paths is not None
        src = pd.read_csv(paths[0])
        det = pd.read_csv(paths[1])
        src_t = src["timestamp_s"].to_numpy(float)
        tau = t["smooth_tau_s"]
        src_v = ema_1d(src_t, src["valence"].to_numpy(float), tau)
        src_a = ema_1d(src_t, src["arousal"].to_numpy(float), tau)
        dv = sample_detected(src_t, det["timestamp_s"].to_numpy(float), det["valence_smooth"].to_numpy(float), lag)
        da = sample_detected(src_t, det["timestamp_s"].to_numpy(float), det["arousal_smooth"].to_numpy(float), lag)
        m = np.isfinite(dv) & np.isfinite(da)
        # debias for overlay readability
        dv_b = dv[m] - np.mean(dv[m] - src_v[m])
        da_b = da[m] - np.mean(da[m] - src_a[m])
        # z-score both for shape overlay (right panel style: dual axis avoided — use z)
        def z(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-12)

        tt = src_t[m]
        axes[i, 0].plot(tt, src_v[m], label="src smooth", color="#4C72B0", lw=1.4)
        axes[i, 0].plot(tt, dv_b, label="det debiased", color="#DD8452", lw=1.2)
        axes[i, 0].set_ylabel("Valence")
        axes[i, 0].set_title(f"{t['trial_id']} [{t.get('category')}] lag={lag:+.2f}s (debiased)")
        axes[i, 0].legend(fontsize=7, loc="upper right")
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].plot(tt, z(src_v[m]), label="src z", color="#4C72B0", lw=1.4)
        axes[i, 1].plot(tt, z(dv[m]), label="det z", color="#DD8452", lw=1.2)
        axes[i, 1].set_ylabel("z-score")
        axes[i, 1].set_title("Shape match (z-scored)")
        axes[i, 1].legend(fontsize=7, loc="upper right")
        axes[i, 1].grid(True, alpha=0.25)
        axes[i, 1].set_ylim(-3.5, 3.5)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "c1_reanalysis_curves.png")
    plt.close(fig)

    # 3) Trial-mean scatter
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    cats = [t.get("category") or "?" for t in trials]
    sx = [t["trial_mean"]["src_V"] for t in trials]
    dx = [t["trial_mean"]["det_V"] for t in trials]
    colors = {"happy": "#2ca02c", "sad": "#1f77b4", "angry": "#d62728", "fear": "#9467bd",
              "surprise": "#ff7f0e", "disgust": "#8c564b", "neutral": "#7f7f7f"}
    for c, x, y, tid in zip(cats, sx, dx, [t["trial_id"] for t in trials]):
        ax.scatter(x, y, c=colors.get(c, "#333"), s=60, label=c if c not in ax.get_legend_handles_labels()[1] else "")
        ax.annotate(tid, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    if len(sx) >= 2:
        r = pearson(np.array(sx), np.array(dx))
        ax.set_title(f"Trial-mean valence (r={r:.3f})")
    ax.set_xlabel("Source mean V")
    ax.set_ylabel("Detected mean V")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "c1_reanalysis_trial_means.png")
    plt.close(fig)


def write_markdown(trials: List[dict], summary: dict, out_path: Path) -> None:
    lines = [
        "# C1 Mirror 再分析\n\n",
        "校正因素：**时间滞后**、**source EMA 平滑（匹配实机低频）**、**去系统偏差**、**仿射幅度校准**。\n\n",
        f"- trials: {summary['n_trials']}\n",
        f"- lag mode: {summary['lag_mode']}\n",
        f"- source smooth τ: {summary['smooth_tau_s']} s\n",
        f"- mean best lag: {summary['mean_best_lag_s']:.3f} s\n\n",
        "## 分层结果（均值）\n\n",
        "| Layer | 含义 | V Pearson | V CCC | V MAE | A Pearson | A CCC | A MAE |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    labels = {
        "L0_raw_150ms": "原始 +150ms",
        "L1_lag": "最优/固定滞后",
        "L2_lag_smooth_src": "+ source 平滑",
        "L3_lag_smooth_debias": "+ 去偏差",
        "L4_lag_smooth_affine": "+ 仿射校准（事后）",
    }
    for lid, lab in labels.items():
        m = summary["layers_mean"][lid]
        lines.append(
            f"| {lid} | {lab} | {m['valence']['pearson_r']:.3f} | {m['valence']['ccc']:.3f} | "
            f"{m['valence']['mae']:.3f} | {m['arousal']['pearson_r']:.3f} | "
            f"{m['arousal']['ccc']:.3f} | {m['arousal']['mae']:.3f} |\n"
        )
    lines.append("\n## 逐 trial\n\n")
    lines.append(
        "| id | cat | lag(s) | L0 V r | L2 V r | L3 V r | L3 A r | L4 V CCC | biasV | biasA |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for t in trials:
        L0 = t["layers"]["L0_raw_150ms"]
        L2 = t["layers"]["L2_lag_smooth_src"]
        L3 = t["layers"]["L3_lag_smooth_debias"]
        L4 = t["layers"]["L4_lag_smooth_affine"]
        lines.append(
            f"| {t['trial_id']} | {t.get('category')} | {t['used_lag_s']:+.2f} | "
            f"{L0['valence']['pearson_r']:.3f} | {L2['valence']['pearson_r']:.3f} | "
            f"{L3['valence']['pearson_r']:.3f} | {L3['arousal']['pearson_r']:.3f} | "
            f"{L4['valence']['ccc']:.3f} | {L3['bias_removed_V']:+.3f} | {L3['bias_removed_A']:+.3f} |\n"
        )
    lines.append(
        "\n## 解读\n\n"
        "- **主指标建议用 L2/L3 的 Pearson r**：已校正延迟与带宽，去偏不改变相关。\n"
        "- **L4 CCC**：仅在仿射校准后评估幅度一致性，须在文中标明 post-hoc。\n"
        "- 系统偏差：valence/arousal 的 `bias_removed_*` 为 det−src 均值。\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=C1_DIR / "result" / "legacy" / "recordings")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--smooth-tau",
        type=float,
        default=0.6,
        help="EMA time constant (s) applied to source VA to match execution smoothing.",
    )
    parser.add_argument(
        "--fixed-lag",
        type=float,
        default=None,
        help="If set, use this lag (s) for all trials instead of per-trial best.",
    )
    parser.add_argument("--lag-min", type=float, default=-1.0)
    parser.add_argument("--lag-max", type=float, default=2.5)
    parser.add_argument("--lag-step", type=float, default=0.05)
    args = parser.parse_args()

    lag_grid = np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step)
    trials: List[dict] = []
    for trial_dir in sorted(args.recordings.iterdir()):
        if not trial_dir.is_dir():
            continue
        paths = find_trial_paths(trial_dir)
        if paths is None:
            continue
        src_path, video_va_path, sess = paths
        print(f"[C1-re] {trial_dir.name}  video_va={video_va_path.parent.name}")
        trials.append(
            analyze_trial(
                trial_dir.name,
                src_path,
                video_va_path,
                sess,
                lag_grid=lag_grid,
                smooth_tau_s=args.smooth_tau,
                fixed_lag_s=args.fixed_lag,
            )
        )

    if not trials:
        print("No trials with source_va + mirror/video_va found.")
        return

    layer_ids = list(trials[0]["layers"].keys())
    layers_mean: Dict[str, Any] = {}
    for lid in layer_ids:
        layers_mean[lid] = {
            "valence": {
                "pearson_r": mean_layer(trials, lid, "valence", "pearson_r"),
                "ccc": mean_layer(trials, lid, "valence", "ccc"),
                "mae": mean_layer(trials, lid, "valence", "mae"),
            },
            "arousal": {
                "pearson_r": mean_layer(trials, lid, "arousal", "pearson_r"),
                "ccc": mean_layer(trials, lid, "arousal", "ccc"),
                "mae": mean_layer(trials, lid, "arousal", "mae"),
            },
        }

    sx = np.array([t["trial_mean"]["src_V"] for t in trials])
    dx = np.array([t["trial_mean"]["det_V"] for t in trials])
    sa = np.array([t["trial_mean"]["src_A"] for t in trials])
    da = np.array([t["trial_mean"]["det_A"] for t in trials])

    summary = {
        "n_trials": len(trials),
        "lag_mode": "fixed" if args.fixed_lag is not None else "per_trial_best",
        "fixed_lag_s": args.fixed_lag,
        "smooth_tau_s": args.smooth_tau,
        "mean_best_lag_s": float(np.mean([t["best_lag_s"] for t in trials])),
        "mean_used_lag_s": float(np.mean([t["used_lag_s"] for t in trials])),
        "layers_mean": layers_mean,
        "trial_mean_corr": {
            "valence": pearson(sx, dx),
            "arousal": pearson(sa, da),
        },
        "trial_ids": [t["trial_id"] for t in trials],
    }

    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten CSV
    rows = []
    for t in trials:
        for lid, layer in t["layers"].items():
            for dim in ("valence", "arousal"):
                rows.append(
                    {
                        "trial_id": t["trial_id"],
                        "category": t.get("category"),
                        "layer": lid,
                        "metric": dim,
                        "lag_s": layer.get("lag_s", t["used_lag_s"]),
                        "smooth_tau_s": t["smooth_tau_s"],
                        **layer[dim],
                    }
                )
    trials_csv = args.result_dir / "c1_reanalysis_trials.csv"
    pd.DataFrame(rows).to_csv(trials_csv, index=False)

    # Compact JSON (drop bulky plot arrays)
    compact_trials = []
    for t in trials:
        ct = {k: v for k, v in t.items() if k != "aligned_for_plot"}
        compact_trials.append(ct)
    summary_path = args.result_dir / "c1_reanalysis_summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "trials": compact_trials}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = args.out_dir / "c1_reanalysis.md"
    write_markdown(trials, summary, md_path)
    write_plots(trials, args.out_dir)

    print("\n=== C1 reanalysis summary ===")
    for lid in layer_ids:
        m = layers_mean[lid]
        print(
            f"{lid:24s}  V r={m['valence']['pearson_r']:+.3f} CCC={m['valence']['ccc']:+.3f}  "
            f"A r={m['arousal']['pearson_r']:+.3f} CCC={m['arousal']['ccc']:+.3f}"
        )
    print(f"trial-mean corr V={summary['trial_mean_corr']['valence']:+.3f} "
          f"A={summary['trial_mean_corr']['arousal']:+.3f}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {trials_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
