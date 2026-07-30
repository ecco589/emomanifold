#!/usr/bin/env python3
"""C1 事件级分析：句内情感转折的物理执行验证（论文主 claim 用）。

视角转换：不问「每帧数值是否重合」，而问论文核心主张——
  当 LLM 内部指令出现【句内大幅情感转折】时，物理面部是否朝正确方向变化？
  （话轮级基线在句内恒定，按构造无法通过此检验。）

三项检验：
  1. 事件级 2AFC：源 VA 在 2 s 窗内摆动超过阈值时，检测 VA 变化方向的命中率
  2. 方差分解：token 级轨迹相对「话轮级（每句一个表情）」额外解释的面部方差
  3. 循环移位替代数据检验：逐 trial 相关的非参显著性 + Fisher 合并 p

输出：
  result/legacy/event/c1_event_analysis.json
  output/legacy/event/c1_event_analysis.md
  output/legacy/event/c1_event_direction.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parent
C1_DIR = CODE_DIR.parent


def ema_1d(t: np.ndarray, y: np.ndarray, tau_s: float) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        dt = max(1e-6, float(t[i] - t[i - 1]))
        alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out


def load_trials(recordings: Path, smooth_tau: float) -> list[dict]:
    trials = []
    for tdir in sorted(recordings.iterdir()):
        if not (tdir / "source_va.csv").is_file():
            continue
        runs = sorted(tdir.glob("mirror/*/video_va.csv"))
        if not runs:
            continue
        src = pd.read_csv(tdir / "source_va.csv")
        det = pd.read_csv(runs[-1])
        sess_path = tdir / "session.json"
        sess = json.loads(sess_path.read_text(encoding="utf-8")) if sess_path.is_file() else {}
        st = src["timestamp_s"].to_numpy(float)
        trials.append(
            dict(
                id=tdir.name,
                cat=sess.get("category"),
                st=st,
                sv=ema_1d(st, src["valence"].to_numpy(float), smooth_tau),
                sa=ema_1d(st, src["arousal"].to_numpy(float), smooth_tau),
                dt=det["timestamp_s"].to_numpy(float),
                dv=det["valence_smooth"].to_numpy(float),
                da=det["arousal_smooth"].to_numpy(float),
            )
        )
    return trials


def best_lag(tr: dict, lag_grid: np.ndarray) -> float:
    best = (-9.0, 0.15)
    for lag in lag_grid:
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


def swing_events(trials: list[dict], dim: str, window_s: float) -> pd.DataFrame:
    """All (source swing, detected swing) pairs over `window_s` horizons."""
    skey, dkey = ("sv", "DV") if dim == "valence" else ("sa", "DA")
    rows = []
    for tr in trials:
        m = np.isfinite(tr[dkey])
        st, s, d = tr["st"][m], tr[skey][m], tr[dkey][m]
        for a in range(len(st)):
            b = int(np.searchsorted(st, st[a] + window_s))
            if b >= len(st):
                break
            rows.append((tr["id"], float(s[b] - s[a]), float(d[b] - d[a])))
    return pd.DataFrame(rows, columns=["id", "d_src", "d_det"])


def direction_accuracy(df: pd.DataFrame, thr: float) -> dict:
    big = df[df["d_src"].abs() > thr]
    if len(big) == 0:
        return {"threshold": thr, "n": 0}
    hit = int((np.sign(big["d_src"]) == np.sign(big["d_det"])).sum())
    p = stats.binomtest(hit, len(big), 0.5, alternative="greater").pvalue
    return {"threshold": thr, "n": int(len(big)), "accuracy": hit / len(big), "p_binomial": float(p)}


def variance_decomposition(trials: list[dict], dim: str) -> dict:
    """R^2 of turn-level (trial mean) vs token-level (per-trial affine of source traj)."""
    skey, dkey = ("sv", "DV") if dim == "valence" else ("sa", "DA")
    pooled = np.concatenate([tr[dkey][np.isfinite(tr[dkey])] for tr in trials])
    gm = pooled.mean()
    ss_tot = ss_turn = ss_token = 0.0
    for tr in trials:
        m = np.isfinite(tr[dkey])
        d, s = tr[dkey][m], tr[skey][m]
        ss_tot += float(((d - gm) ** 2).sum())
        ss_turn += float(((d - d.mean()) ** 2).sum())
        X = np.vstack([s, np.ones_like(s)]).T
        coef = np.linalg.lstsq(X, d, rcond=None)[0]
        ss_token += float(((d - X @ coef) ** 2).sum())
    r2_turn = 1 - ss_turn / ss_tot
    r2_token = 1 - ss_token / ss_tot
    return {
        "R2_turn_level": r2_turn,
        "R2_token_level": r2_token,
        "R2_gain": r2_token - r2_turn,
        "residual_explained": (r2_token - r2_turn) / (1 - r2_turn),
    }


def surrogate_test(trials: list[dict], dim: str, n_perm: int, rng: np.random.Generator) -> dict:
    skey, dkey = ("sv", "DV") if dim == "valence" else ("sa", "DA")
    pvals = []
    skipped = 0
    for tr in trials:
        m = np.isfinite(tr[dkey])
        s, d = tr[skey][m], tr[dkey][m]
        if len(d) < 12:
            skipped += 1
            continue
        robs = np.corrcoef(s, d)[0, 1]
        hi = len(d) - 5
        if hi <= 5:
            skipped += 1
            continue
        null = [
            np.corrcoef(s, np.roll(d, int(rng.integers(5, hi))))[0, 1]
            for _ in range(n_perm)
        ]
        pvals.append(float((np.sum(np.array(null) >= robs) + 1) / (n_perm + 1)))
    if not pvals:
        return {
            "per_trial_p": [],
            "n_significant": 0,
            "n_trials": 0,
            "n_skipped_short": skipped,
            "fisher_combined_p": None,
        }
    chi = -2 * np.sum(np.log(pvals))
    fisher_p = float(stats.chi2.sf(chi, 2 * len(pvals)))
    return {
        "per_trial_p": pvals,
        "n_significant": int(sum(p < 0.05 for p in pvals)),
        "n_trials": len(pvals),
        "n_skipped_short": skipped,
        "fisher_combined_p": fisher_p,
    }


def plot_events(dfv: pd.DataFrame, dfa: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for ax, df, name, thr in ((axes[0], dfv, "Valence", 0.5), (axes[1], dfa, "Arousal", 0.3)):
        big = df[df["d_src"].abs() > thr]
        small = df[df["d_src"].abs() <= thr]
        ax.scatter(small["d_src"], small["d_det"], s=6, alpha=0.15, color="#999999",
                   label=f"|Δsrc| ≤ {thr}")
        hit = big[np.sign(big["d_src"]) == np.sign(big["d_det"])]
        miss = big[np.sign(big["d_src"]) != np.sign(big["d_det"])]
        ax.scatter(hit["d_src"], hit["d_det"], s=14, alpha=0.7, color="#2ca02c",
                   label="direction match")
        ax.scatter(miss["d_src"], miss["d_det"], s=14, alpha=0.8, color="#d62728",
                   label="mismatch")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        acc = len(hit) / max(1, len(big))
        ax.set_title(f"{name}: commanded swing vs facial response\n"
                     f"|Δsrc|>{thr}: {acc:.0%} match (n={len(big)})")
        ax.set_xlabel("Commanded change over 2 s (source)")
        ax.set_ylabel("Facial change over 2 s (detected)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)
    fig.suptitle("C1: intra-utterance affect transitions are physically executed")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=C1_DIR / "result" / "legacy" / "recordings")
    parser.add_argument("--out-dir", type=Path, default=C1_DIR / "output" / "legacy" / "event")
    parser.add_argument("--result-dir", type=Path, default=C1_DIR / "result" / "legacy" / "event")
    parser.add_argument("--window", type=float, default=2.0, help="Swing window (s).")
    parser.add_argument("--smooth-tau", type=float, default=0.6)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trials = load_trials(args.recordings, args.smooth_tau)
    print(f"[C1-ev] loaded {len(trials)} trials")

    lag_grid = np.arange(-1.0, 2.51, 0.05)
    for tr in trials:
        lag = best_lag(tr, lag_grid)
        tr["lag"] = lag
        tr["DV"] = np.interp(tr["st"] + lag, tr["dt"], tr["dv"], left=np.nan, right=np.nan)
        tr["DA"] = np.interp(tr["st"] + lag, tr["dt"], tr["da"], left=np.nan, right=np.nan)
        print(f"[C1-ev] {tr['id']} [{tr['cat']}] lag={lag:+.2f}s")

    dfv = swing_events(trials, "valence", args.window)
    dfa = swing_events(trials, "arousal", args.window)

    results = {
        "n_trials": len(trials),
        "window_s": args.window,
        "smooth_tau_s": args.smooth_tau,
        "mean_lag_s": float(np.mean([tr["lag"] for tr in trials])),
        "event_2afc": {
            "valence": [direction_accuracy(dfv, thr) for thr in (0.3, 0.5, 0.8)],
            "arousal": [direction_accuracy(dfa, thr) for thr in (0.2, 0.3)],
        },
        "variance_decomposition": {
            "valence": variance_decomposition(trials, "valence"),
            "arousal": variance_decomposition(trials, "arousal"),
        },
        "surrogate": {
            "valence": surrogate_test(trials, "valence", args.n_perm, rng),
            "arousal": surrogate_test(trials, "arousal", args.n_perm, rng),
        },
    }

    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "c1_event_analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_events(dfv, dfa, args.out_dir / "c1_event_direction.png")

    md = ["# C1 事件级分析（句内转折的物理执行）\n\n"]
    md.append(f"- trials: {len(trials)}，swing 窗口 {args.window}s，平均滞后 "
              f"{results['mean_lag_s']:.2f}s\n\n")
    md.append("## 1. 指令转折 → 面部响应方向（2AFC，随机基线 50%）\n\n")
    md.append("| 维度 | 阈值 | n | 方向命中 | binomial p |\n|---|---:|---:|---:|---:|\n")
    for dim in ("valence", "arousal"):
        for e in results["event_2afc"][dim]:
            if e.get("n", 0) == 0:
                continue
            md.append(f"| {dim} | {e['threshold']} | {e['n']} | "
                      f"{e['accuracy']:.1%} | {e['p_binomial']:.2e} |\n")
    md.append("\n## 2. 方差分解：token 级 vs 话轮级\n\n")
    md.append("| 维度 | R² 话轮级 | R² token级 | 增益 | 残差解释率 |\n|---|---:|---:|---:|---:|\n")
    for dim in ("valence", "arousal"):
        v = results["variance_decomposition"][dim]
        md.append(f"| {dim} | {v['R2_turn_level']:.3f} | {v['R2_token_level']:.3f} | "
                  f"+{v['R2_gain']:.3f} | {v['residual_explained']:.1%} |\n")
    md.append("\n## 3. 循环移位替代数据检验\n\n")
    for dim in ("valence", "arousal"):
        s = results["surrogate"][dim]
        md.append(f"- {dim}: {s['n_significant']}/{s['n_trials']} trials p<0.05，"
                  f"Fisher 合并 p = {s['fisher_combined_p']:.2e}\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "c1_event_analysis.md").write_text("".join(md), encoding="utf-8")

    print("\n=== headline numbers ===")
    for dim in ("valence", "arousal"):
        for e in results["event_2afc"][dim]:
            if e.get("n", 0):
                print(f"{dim} swing>{e['threshold']}: {e['accuracy']:.1%} of {e['n']} "
                      f"(p={e['p_binomial']:.1e})")
    for dim in ("valence", "arousal"):
        v = results["variance_decomposition"][dim]
        print(f"{dim} R2 turn={v['R2_turn_level']:.3f} -> token={v['R2_token_level']:.3f}")
    print(f"Wrote {args.result_dir / 'c1_event_analysis.json'}")
    print(f"Wrote {args.out_dir / 'c1_event_analysis.md'}")
    print(f"Wrote {args.out_dir / 'c1_event_direction.png'}")


if __name__ == "__main__":
    main()
