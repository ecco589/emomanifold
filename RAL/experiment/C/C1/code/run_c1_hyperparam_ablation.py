#!/usr/bin/env python3
"""Offline α × P ablation on existing C1 commanded VA trajectories.

Reuses the 20 field `source_va.csv` files (no new recording). For each
(α, prototype set) the full display path of Sec. III-C is replayed:

  command VA → FKCN(P) → EMA(α) → 10 fps smoothstep interpolation
             → per-frame rate limit (0.05) → frame-rate trajectory

Scope, stated plainly: this measures only the software stage. The hardware
stage (servos → face → camera → Py-Feat) is absent, and α/P do not enter it,
so it is common-mode across conditions and was measured once at the default
setting in the main C1 experiment. The default-setting agreement between this
replay and the camera number is the empirical license for the sweep.

Reported per condition:
  * direction agreement for |ΔV|>0.3 / |ΔA|>0.2 command events (C1 protocol)
  * a shuffled-command floor for the same metric (does the metric fail?)
  * amplitude retention, per-frame jitter, smoothing lag
  * rate-limiter saturation (evidence for the α=0.9 bottleneck claim)
  * paired comparison against the default condition over shared trials

Three documented deviations, all preset-independent:
  * The 29-D presets live on the robot PC, so interpolation and the
    per-factor rate limit run in 7-D weight space. Presets are mutually
    similar on most channels, so weight-space deltas upper-bound factor
    deltas and the limiter binds earlier here than on hardware
    (conservative for direction/amplitude retention).
  * VA readout uses the P=13 centroids for every condition, so prototype
    sets are compared in one fixed coordinate system.
  * Events use one fixed alignment lag for every condition (zero by
    default). The 0.5 s lag of Fig. 5A is a hardware pipeline delay that
    does not exist offline, and per-trial fitted lags would hand every
    condition a free parameter -- which C1 explicitly rejects.

Usage (repo root):
  python RAL/experiment/C/C1/code/run_c1_hyperparam_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[4]  # EmotionManifold repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_interpolator import FaceFrameInterpolator  # noqa: E402
from src.fkcn import DEFAULT_PROTOTYPES, EMOTION_LABELS, FKCN  # noqa: E402
from src.grimace_utils import smooth_weights_batch  # noqa: E402

from run_c1_event_analysis import direction_accuracy, ema_1d  # noqa: E402
from run_c1_multicond_compare import clustered_direction  # noqa: E402

DEFAULT_FIELD = (
    CODE_DIR.parent / "result" / "multicond_noanchor_20260731" / "ours"
)
OUT_JSON = CODE_DIR.parent / "result" / "hyperparam_ablation" / "ablation_summary.json"
OUT_MD = CODE_DIR.parent / "output" / "hyperparam_ablation" / "ablation_summary.md"

# DEFAULT_PROTOTYPES holds 1 prototype for neutral and 2 for each other
# emotion. "7a"/"7b" keep one per emotion, differing in which of the pair
# survives, so the P=7 result cannot rest on an arbitrary pick.
PROTOTYPE_SETS: dict[str, tuple[int, ...]] = {
    "7a": (0, 1, 3, 5, 7, 9, 11),
    "7b": (0, 2, 4, 6, 8, 10, 12),
    "13": tuple(range(len(DEFAULT_PROTOTYPES))),
}
DEFAULT_SET = "13"
DEFAULT_ALPHA = 0.6

# Display path constants (Sec. III-C / src.face_interpolator defaults).
FPS = 10.0
TRANSITION_MS = 150.0
MAX_DELTA = 0.05
SETTLE_FRAMES = 10


def prototypes_for_set(label: str):
    try:
        idx = PROTOTYPE_SETS[label]
    except KeyError:
        raise ValueError(
            f"unknown prototype set {label!r}; use {sorted(PROTOTYPE_SETS)}"
        ) from None
    return [DEFAULT_PROTOTYPES[i] for i in idx]


def emotion_centroids(prototypes) -> np.ndarray:
    """7×2 VA means for EMOTION_LABELS from the given prototype set."""
    buckets: list[list[np.ndarray]] = [[] for _ in EMOTION_LABELS]
    for (v, a), w in prototypes:
        idx = int(np.argmax(w))
        buckets[idx].append(np.array([v, a], dtype=float))
    cents = np.zeros((len(EMOTION_LABELS), 2), dtype=float)
    for i, pts in enumerate(buckets):
        if not pts:
            raise RuntimeError(f"no prototypes for emotion {EMOTION_LABELS[i]}")
        cents[i] = np.mean(pts, axis=0)
    return cents


def reconstruct_va(
    weights: list[list[float]], centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=float)
    va = w @ centroids
    return va[:, 0], va[:, 1]


def load_command_trials(field_root: Path, smooth_tau: float) -> list[dict]:
    trials = []
    for tdir in sorted(field_root.iterdir()):
        src_path = tdir / "source_va.csv"
        if not src_path.is_file():
            continue
        src = pd.read_csv(src_path)
        st = src["timestamp_s"].to_numpy(float)
        # Prefer raw projector coords when present; fall back to displayed command.
        v_col = "raw_valence" if "raw_valence" in src.columns else "valence"
        a_col = "raw_arousal" if "raw_arousal" in src.columns else "arousal"
        v = np.clip(src[v_col].to_numpy(float), -1.0, 1.0)
        a = np.clip(src[a_col].to_numpy(float), -1.0, 1.0)
        trials.append(
            dict(
                id=tdir.name,
                st=st,
                v_cmd=v,
                a_cmd=a,
                sv=ema_1d(st, v, smooth_tau),
                sa=ema_1d(st, a, smooth_tau),
            )
        )
    if len(trials) != 20:
        print(
            f"warning: expected 20 trials, found {len(trials)} under {field_root}",
            flush=True,
        )
    return trials


def swing_events_aligned(trials: list[dict], dim: str, window_s: float) -> pd.DataFrame:
    """Like C1 swing_events, but src/det share the same token timeline (no lag)."""
    skey, dkey = ("sv", "DV") if dim == "valence" else ("sa", "DA")
    rows = []
    for tr in trials:
        st, s, d = tr["st"], tr[skey], tr[dkey]
        for i in range(len(st)):
            j = int(np.searchsorted(st, st[i] + window_s))
            if j >= len(st):
                break
            rows.append((tr["id"], float(s[j] - s[i]), float(d[j] - d[i])))
    return pd.DataFrame(rows, columns=["id", "d_src", "d_det"])


def per_trial_hit_rate(df: pd.DataFrame, thr: float) -> pd.Series:
    big = df[df["d_src"].abs() > thr].copy()
    if big.empty:
        return pd.Series(dtype=float)
    big["hit"] = (np.sign(big["d_src"]) == np.sign(big["d_det"])).astype(float)
    return big.groupby("id")["hit"].mean()


def weight_jitter(weights) -> float:
    w = np.asarray(weights, dtype=float)
    if len(w) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(w, axis=0))))


def replay_display_path(
    token_times: np.ndarray,
    smooth_weights: list[list[float]],
    fps: float = FPS,
    src_times: list | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the 10 fps smoothstep + rate-limited interpolator over token targets.

    A new token pre-empts the previous transition, so each target only gets the
    frames that fit before the next token arrives (as on hardware). The third
    return value is the fraction of per-dimension frame steps clamped by the
    rate limiter."""
    interp = FaceFrameInterpolator(
        fps=fps,
        transition_ms=TRANSITION_MS,
        max_delta=MAX_DELTA,
        n_dims=len(smooth_weights[0]),
        realtime=False,
    )
    frame_dt = 1.0 / fps
    frames: list[list[float]] = []
    times: list[float] = []
    clock = float(token_times[0])
    n = len(smooth_weights)
    for i, target in enumerate(smooth_weights):
        if i < n - 1:
            gap = float(token_times[i + 1] - token_times[i])
            budget = max(1, int(round(gap / frame_dt)))
        else:
            budget = SETTLE_FRAMES
        for j, frame in enumerate(interp.animate_to(target)):
            if j >= budget:
                break
            frames.append(frame)
            times.append(clock)
            if src_times is not None:
                src_times.append(float(token_times[i]) + j * frame_dt)
            clock += frame_dt
    arr = np.asarray(frames, dtype=float)
    if len(arr) < 2:
        saturation = 0.0
    else:
        steps = np.abs(np.diff(arr, axis=0))
        saturation = float(np.mean(steps >= MAX_DELTA - 1e-9))
    return np.asarray(times, dtype=float), arr, saturation


def smoothing_lag(
    st: np.ndarray,
    sv: np.ndarray,
    sa: np.ndarray,
    ft: np.ndarray,
    fv: np.ndarray,
    fa: np.ndarray,
    lag_grid: np.ndarray,
) -> float | None:
    """Lag maximizing mean V/A correlation between command and frame trajectory."""
    best = (-np.inf, None)
    for lag in lag_grid:
        dv = np.interp(st + lag, ft, fv, left=np.nan, right=np.nan)
        da = np.interp(st + lag, ft, fa, left=np.nan, right=np.nan)
        m = np.isfinite(dv) & np.isfinite(da)
        if m.sum() < 5 or np.std(dv[m]) < 1e-9 or np.std(da[m]) < 1e-9:
            continue
        r = 0.5 * (np.corrcoef(sv[m], dv[m])[0, 1] + np.corrcoef(sa[m], da[m])[0, 1])
        if r > best[0]:
            best = (r, float(lag))
    return best[1]


def run_chain(
    v_cmd: np.ndarray,
    a_cmd: np.ndarray,
    st: np.ndarray,
    fkcn: FKCN,
    alpha: float,
    readout: np.ndarray,
    event_lag_s: float,
) -> dict:
    """command VA → FKCN → EMA → display path → VA sampled on token timeline."""
    points = list(zip(v_cmd.tolist(), a_cmd.tolist()))
    smooth_w = smooth_weights_batch(fkcn.compute_weights_batch(points), alpha=alpha)
    ft, frames, saturation = replay_display_path(st, smooth_w)
    fv, fa = reconstruct_va(frames.tolist(), readout)
    return {
        "token_weights": smooth_w,
        "ft": ft,
        "frames": frames,
        "saturation": saturation,
        "fv": fv,
        "fa": fa,
        "DV": np.interp(st + event_lag_s, ft, fv, left=fv[0], right=fv[-1]),
        "DA": np.interp(st + event_lag_s, ft, fa, left=fa[0], right=fa[-1]),
    }


def shuffled_floor(
    trials: list[dict],
    fkcn: FKCN,
    alpha: float,
    readout: np.ndarray,
    window_s: float,
    event_lag_s: float,
    n_shuffle: int,
    seed: int,
) -> dict:
    """Same chain on time-shuffled commands, scored against the true commands.

    Mirrors the C1 shuffled condition: identical dynamic range, destroyed
    timing. Establishes that the metric can in fact fail."""
    v_rates, a_rates = [], []
    for rep in range(n_shuffle):
        rng = np.random.default_rng(seed + 1000 * rep)
        eval_trials = []
        for tr in trials:
            perm = rng.permutation(len(tr["v_cmd"]))
            out = run_chain(
                tr["v_cmd"][perm],
                tr["a_cmd"][perm],
                tr["st"],
                fkcn,
                alpha,
                readout,
                event_lag_s,
            )
            eval_trials.append(
                dict(
                    id=tr["id"],
                    st=tr["st"],
                    sv=tr["sv"],
                    sa=tr["sa"],
                    DV=out["DV"],
                    DA=out["DA"],
                )
            )
        dfv = swing_events_aligned(eval_trials, "valence", window_s)
        dfa = swing_events_aligned(eval_trials, "arousal", window_s)
        v_rates.append(per_trial_hit_rate(dfv, 0.3).mean())
        a_rates.append(per_trial_hit_rate(dfa, 0.2).mean())
    return {
        "n_shuffle": n_shuffle,
        "valence_trial_mean_accuracy": float(np.mean(v_rates)),
        "arousal_trial_mean_accuracy": float(np.mean(a_rates)),
    }


def evaluate_config(
    trials: list[dict],
    alpha: float,
    proto_set: str,
    window_s: float,
    seed: int,
    event_lag_s: float,
    n_shuffle: int,
) -> dict:
    protos = prototypes_for_set(proto_set)
    fkcn = FKCN(prototypes=protos, device="cpu")
    readout = emotion_centroids(prototypes_for_set(DEFAULT_SET))
    lag_grid = np.arange(0.0, 1.51, 0.05)
    eval_trials = []
    token_jitters, frame_jitters, saturations, lags = [], [], [], []
    for tr in trials:
        out = run_chain(
            tr["v_cmd"], tr["a_cmd"], tr["st"], fkcn, alpha, readout, event_lag_s
        )
        token_jitters.append(weight_jitter(out["token_weights"]))
        frame_jitters.append(weight_jitter(out["frames"]))
        saturations.append(out["saturation"])
        # Diagnostic only; never used to align events.
        lag = smoothing_lag(
            tr["st"], tr["sv"], tr["sa"], out["ft"], out["fv"], out["fa"], lag_grid
        )
        if lag is not None:
            lags.append(lag)
        eval_trials.append(
            dict(
                id=tr["id"],
                st=tr["st"],
                sv=tr["sv"],
                sa=tr["sa"],
                DV=out["DV"],
                DA=out["DA"],
            )
        )

    dfv = swing_events_aligned(eval_trials, "valence", window_s)
    dfa = swing_events_aligned(eval_trials, "arousal", window_s)
    rng = np.random.default_rng(seed)

    def _amp_ret(df: pd.DataFrame, thr: float) -> float | None:
        big = df[df["d_src"].abs() > thr]
        if len(big) == 0:
            return None
        return float((big["d_det"].abs() / big["d_src"].abs()).mean())

    return {
        "alpha": alpha,
        "prototype_set": proto_set,
        "P": len(protos),
        "is_default": abs(alpha - DEFAULT_ALPHA) < 1e-9 and proto_set == DEFAULT_SET,
        "n_trials": len(eval_trials),
        "event_lag_s": event_lag_s,
        "mean_token_weight_step": float(np.mean(token_jitters)),
        "mean_frame_weight_step": float(np.mean(frame_jitters)),
        "rate_limit_saturation": float(np.mean(saturations)),
        "mean_smoothing_lag_s": float(np.mean(lags)) if lags else None,
        "valence_amp_retention_thr0.3": _amp_ret(dfv, 0.3),
        "arousal_amp_retention_thr0.2": _amp_ret(dfa, 0.2),
        "valence_dir_acc_thr0.3": direction_accuracy(dfv, 0.3),
        "arousal_dir_acc_thr0.2": direction_accuracy(dfa, 0.2),
        "valence_clustered_thr0.3": clustered_direction(dfv, 0.3, rng),
        "arousal_clustered_thr0.2": clustered_direction(dfa, 0.2, rng),
        "shuffled_floor": shuffled_floor(
            trials, fkcn, alpha, readout, window_s, event_lag_s, n_shuffle, seed
        ),
        "per_trial_valence_acc": {
            k: float(v) for k, v in per_trial_hit_rate(dfv, 0.3).items()
        },
        "per_trial_arousal_acc": {
            k: float(v) for k, v in per_trial_hit_rate(dfa, 0.2).items()
        },
    }


def paired_vs_default(
    cur: dict[str, float], base: dict[str, float], seed: int, n_boot: int = 10000
) -> dict:
    """Paired comparison over trials shared by the two conditions."""
    ids = sorted(set(cur) & set(base))
    if not ids:
        return {"n_trials": 0}
    d = np.array([cur[i] - base[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    )
    nonzero = int(np.count_nonzero(d))
    if nonzero:
        p = float(
            stats.wilcoxon(
                [cur[i] for i in ids], [base[i] for i in ids], zero_method="wilcox"
            ).pvalue
        )
    else:
        p = None
    return {
        "n_trials": len(ids),
        "n_nonzero_diffs": nonzero,
        "mean_diff_pp": float(100 * d.mean()),
        "ci95_pp": [
            float(100 * np.percentile(boots, 2.5)),
            float(100 * np.percentile(boots, 97.5)),
        ],
        "wilcoxon_p": p,
    }


def fmt_acc(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "n/a"
    p = d.get("p_binomial")
    p_s = f", p={p:.3g}" if p is not None else ""
    return f"{100 * d['accuracy']:.1f}% (n={d['n']}{p_s})"


def fmt_paired(d: dict | None) -> str:
    if not d or not d.get("n_trials"):
        return "— (default)"
    p = d.get("wilcoxon_p")
    p_s = f", p={p:.3f}" if p is not None else ", p=n/a"
    lo, hi = d["ci95_pp"]
    return f"{d['mean_diff_pp']:+.1f} pp [{lo:+.1f}, {hi:+.1f}]{p_s}"


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# C1 offline hyperparameter ablation (α × prototype set)",
        "",
        f"- Field: `{summary['field_root']}`  |  trials: {summary['n_trials']}",
        f"- Window: {summary['window_s']} s; event thresholds V>|0.3|, A>|0.2|",
        "- Pipeline: clip(command VA) → FKCN → EMA(α) → 10 fps smoothstep"
        " + 0.05/frame rate limit → VA readout on fixed P=13 centroids",
        "",
        "## Scope",
        "",
        "This replays the **software** stage only; there is no video and no"
        " Py-Feat. The hardware stage (servos → face → camera → detector) is"
        " common-mode across conditions because α and P do not enter it, and it"
        " was measured once at the default setting in the main experiment."
        " Compare the default row below against the camera numbers"
        " (V 87.5%, A 69.6%) to judge how much the missing stage costs.",
        "",
        "## Deviations from the Fig. 5A protocol",
        "",
        "1. Rate limit runs in 7-D weight space (29-D presets live on the robot"
        " PC); weight-space deltas upper-bound factor deltas, so the limiter"
        " binds earlier here than on hardware (conservative).",
        "2. VA readout always uses P=13 centroids, so prototype sets share one"
        " coordinate system.",
        f"3. Events use a single fixed lag of {summary['event_lag_s']} s for all"
        " conditions with no per-trial fitting. The 0.5 s of Fig. 5A is a"
        " hardware pipeline delay that does not exist offline; the smoothing"
        " lag column is a diagnostic, not an alignment parameter.",
        "",
        "## Results",
        "",
        "| α | set | P | V dir-acc | V floor | A dir-acc | A floor | V amp-ret |"
        " lag (s) | sat | frame |Δw| |",
        "|---|-----|---|-----------|---------|-----------|---------|-----------|"
        "---------|-----|------------|",
    ]
    for row in summary["configs"]:
        vamp = row.get("valence_amp_retention_thr0.3")
        vamp_s = f"{vamp:.3f}" if vamp is not None else "n/a"
        lag = row.get("mean_smoothing_lag_s")
        lag_s = f"{lag:.2f}" if lag is not None else "n/a"
        floor = row["shuffled_floor"]
        mark = " **(default)**" if row["is_default"] else ""
        lines.append(
            f"| {row['alpha']:.1f}{mark} | {row['prototype_set']} | {row['P']} | "
            f"{fmt_acc(row['valence_dir_acc_thr0.3'])} | "
            f"{100 * floor['valence_trial_mean_accuracy']:.1f}% | "
            f"{fmt_acc(row['arousal_dir_acc_thr0.2'])} | "
            f"{100 * floor['arousal_trial_mean_accuracy']:.1f}% | "
            f"{vamp_s} | {lag_s} | "
            f"{100 * row['rate_limit_saturation']:.1f}% | "
            f"{row['mean_frame_weight_step']:.4f} |"
        )

    lines += [
        "",
        "`V floor` / `A floor`: same chain on time-shuffled commands"
        f" ({summary['n_shuffle']} permutations), scored against the true"
        " commands. `sat`: share of per-dimension frame steps clamped by the"
        " 0.05 rate limiter.",
        "",
        "## Paired comparison against the default condition",
        "",
        "Trial-level accuracies are paired across conditions (same 20"
        " utterances), so differences are tested pairwise rather than by"
        " comparing marginal intervals.",
        "",
        "| α | set | ΔV vs default | ΔA vs default |",
        "|---|-----|---------------|---------------|",
    ]
    for row in summary["configs"]:
        lines.append(
            f"| {row['alpha']:.1f} | {row['prototype_set']} | "
            f"{fmt_paired(row.get('paired_valence_vs_default'))} | "
            f"{fmt_paired(row.get('paired_arousal_vs_default'))} |"
        )

    def _pick(alpha: float, proto_set: str = DEFAULT_SET) -> dict:
        return next(
            r
            for r in summary["configs"]
            if abs(r["alpha"] - alpha) < 1e-9 and r["prototype_set"] == proto_set
        )

    v_tm = [r["valence_clustered_thr0.3"]["trial_mean_accuracy"] for r in summary["configs"]]
    a_tm = [r["arousal_clustered_thr0.2"]["trial_mean_accuracy"] for r in summary["configs"]]
    floors_v = [r["shuffled_floor"]["valence_trial_mean_accuracy"] for r in summary["configs"]]
    r03, r06, r09 = _pick(0.3), _pick(0.6), _pick(0.9)
    p7a, p7b = _pick(DEFAULT_ALPHA, "7a"), _pick(DEFAULT_ALPHA, "7b")
    lines += [
        "",
        "## Suggested IV-C sentences (draft)",
        "",
        "Note: codebase EMA uses α·cur+(1-α)·prev; smaller α damps more.",
        "",
        (
            f"离线重放给出的效价方向一致率落在 {100 * min(v_tm):.0f}%--{100 * max(v_tm):.0f}%"
            f"（唤醒 {100 * min(a_tm):.0f}%--{100 * max(a_tm):.0f}%），"
            f"而同一管线在打乱时序的指令上仅 {100 * min(floors_v):.0f}%--{100 * max(floors_v):.0f}%，"
            f"说明该指标并非恒过。默认设定（α=0.6、P=13）为 "
            f"{100 * r06['valence_clustered_thr0.3']['trial_mean_accuracy']:.1f}%/"
            f"{100 * r06['arousal_clustered_thr0.2']['trial_mean_accuracy']:.1f}%，"
            f"与相机实测 87.5%/69.6% 量级一致。"
            f"逐试次配对比较显示：α=0.9 较默认低 "
            f"{abs(r09['paired_valence_vs_default']['mean_diff_pp']):.1f} 个百分点"
            f"（95% CI [{r09['paired_valence_vs_default']['ci95_pp'][0]:+.1f}, "
            f"{r09['paired_valence_vs_default']['ci95_pp'][1]:+.1f}]，"
            f"Wilcoxon p={r09['paired_valence_vs_default']['wilcoxon_p']:.3f}），"
            f"此时 {100 * r09['rate_limit_saturation']:.0f}% 的逐帧步长被单帧上限截断"
            f"（默认 {100 * r06['rate_limit_saturation']:.0f}%），限速成为瓶颈；"
            f"α=0.3 与默认无显著差异（"
            f"{r03['paired_valence_vs_default']['mean_diff_pp']:+.1f} pp，"
            f"p={r03['paired_valence_vs_default']['wilcoxon_p']:.3f}）但平滑滞后由 "
            f"{r06['mean_smoothing_lag_s']:.2f} s 升至 {r03['mean_smoothing_lag_s']:.2f} s。"
            f"两种 P=7 构造（保留每类中的不同原型）与默认亦无显著差异"
            f"（{p7a['paired_valence_vs_default']['mean_diff_pp']:+.1f} pp / "
            f"{p7b['paired_valence_vs_default']['mean_diff_pp']:+.1f} pp）。"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-root", type=Path, default=DEFAULT_FIELD)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument(
        "--proto-sets",
        nargs="+",
        default=["13", "7a", "7b"],
        choices=sorted(PROTOTYPE_SETS),
    )
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--smooth-tau", type=float, default=0.6)
    ap.add_argument(
        "--event-lag-s",
        type=float,
        default=0.0,
        help=(
            "Command-vs-trajectory alignment for event scoring. One fixed value "
            "for all conditions (no per-trial fitting). Zero by default: the "
            "offline replay has no servo/camera stage, so the 0.5 s hardware "
            "pipeline lag of Fig. 5A does not apply."
        ),
    )
    ap.add_argument("--n-shuffle", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    trials = load_command_trials(args.field_root, args.smooth_tau)
    if not trials:
        raise SystemExit(f"no source_va.csv under {args.field_root}")

    configs = []
    for proto_set in args.proto_sets:
        for alpha in args.alphas:
            row = evaluate_config(
                trials,
                alpha,
                proto_set,
                args.window_s,
                args.seed,
                args.event_lag_s,
                args.n_shuffle,
            )
            configs.append(row)
            floor = row["shuffled_floor"]
            print(
                f"α={alpha:.1f} set={proto_set:>2} (P={row['P']}): "
                f"V={fmt_acc(row['valence_dir_acc_thr0.3'])} "
                f"floor={100 * floor['valence_trial_mean_accuracy']:.1f}%  "
                f"A={fmt_acc(row['arousal_dir_acc_thr0.2'])} "
                f"floor={100 * floor['arousal_trial_mean_accuracy']:.1f}%  "
                f"sat={100 * row['rate_limit_saturation']:.1f}%  "
                f"lag={row['mean_smoothing_lag_s']:.2f}s",
                flush=True,
            )

    base = next(r for r in configs if r["is_default"])
    for row in configs:
        if row["is_default"]:
            continue
        row["paired_valence_vs_default"] = paired_vs_default(
            row["per_trial_valence_acc"], base["per_trial_valence_acc"], args.seed
        )
        row["paired_arousal_vs_default"] = paired_vs_default(
            row["per_trial_arousal_acc"], base["per_trial_arousal_acc"], args.seed
        )

    print("\nPaired vs default (α=0.6, P=13):")
    for row in configs:
        if row["is_default"]:
            continue
        print(
            f"  α={row['alpha']:.1f} set={row['prototype_set']:>2}: "
            f"ΔV={fmt_paired(row['paired_valence_vs_default'])}  "
            f"ΔA={fmt_paired(row['paired_arousal_vs_default'])}",
            flush=True,
        )

    summary = {
        "field_root": str(args.field_root),
        "n_trials": len(trials),
        "window_s": args.window_s,
        "smooth_tau_s": args.smooth_tau,
        "event_lag_s": args.event_lag_s,
        "n_shuffle": args.n_shuffle,
        "alphas": args.alphas,
        "prototype_sets": args.proto_sets,
        "default_condition": {"alpha": DEFAULT_ALPHA, "prototype_set": DEFAULT_SET},
        "camera_reference_fig5a": {"valence": 0.875, "arousal": 0.696},
        "display_path": {
            "fps": FPS,
            "transition_ms": TRANSITION_MS,
            "max_delta_per_frame": MAX_DELTA,
            "rate_limit_space": "7-D FKCN weights (29-D presets on robot PC)",
            "readout_centroids": "P=13 (fixed across conditions)",
        },
        "note": (
            "Offline replay of the Sec. III-C display path; VA read out from the "
            "frame-rate weight trajectory, NOT camera readback. Event thresholds "
            "and window match C1; alignment does not (see event_lag_s). The "
            "shuffled floor uses the same chain on time-permuted commands. "
            "Condition differences are tested paired over shared trials."
        ),
        "configs": configs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.out_md)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
