#!/usr/bin/env python3
"""本场 C1 四条件正式对比（对齐旧 C1 事件分析协议）。

对每个 trial：
  commanded = ours/source_va.csv（真实回复轨迹）
  detected  = 各条件 mirror/*/video_va.csv

这样 Mishra/Xpress（句级固定脸）与 random 都能和「同一条指令轨迹」比方向命中。
另附 ours 自洽：ours source vs ours face（经典 Mirror）。
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
sys.path.insert(0, str(CODE_DIR))
from run_c1_event_analysis import (  # noqa: E402
    best_lag,
    direction_accuracy,
    ema_1d,
    swing_events,
    variance_decomposition,
    surrogate_test,
)


SYSTEMS = ("ours", "mishra_gpt", "xpress_gpt", "random")


def fit_static_calibration(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mirror 对机器人卡通脸的 VA 读数有系统偏差（效价被压缩约 4x 且混入唤醒分量）。
    用静止脸条件（mishra/xpress，每条视频的显示 VA 已知且恒定）拟合
    [det_V, det_A, 1] -> displayed 的线性校准；校准集与被测数据完全独立。"""
    X, yv, ya = [], [], []
    for sysn in ("mishra_gpt", "xpress_gpt"):
        for tdir in sorted((root / sysn).iterdir()):
            if not tdir.is_dir():
                continue
            sess_path = tdir / "session.json"
            if not sess_path.is_file():
                continue
            lab = (json.loads(sess_path.read_text(encoding="utf-8")).get("label") or {})
            runs = sorted(tdir.glob("mirror/*/video_va.csv"))
            if "valence" not in lab or not runs:
                continue
            det = pd.read_csv(runs[-1])
            d = det[det["timestamp_s"] > 1.0]
            X.append([d["valence_smooth"].mean(), d["arousal_smooth"].mean(), 1.0])
            yv.append(lab["valence"])
            ya.append(lab["arousal"])
    X = np.asarray(X)
    cv = np.linalg.lstsq(X, np.asarray(yv), rcond=None)[0]
    ca = np.linalg.lstsq(X, np.asarray(ya), rcond=None)[0]
    return cv, ca


def load_face(
    trial_dir: Path,
    smooth_tau: float,
    calib: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    runs = sorted(trial_dir.glob("mirror/*/video_va.csv"))
    if not runs:
        return None
    det = pd.read_csv(runs[-1])
    sess_path = trial_dir / "session.json"
    sess = json.loads(sess_path.read_text(encoding="utf-8")) if sess_path.is_file() else {}
    dt = det["timestamp_s"].to_numpy(float)
    dv = det["valence_smooth"].to_numpy(float)
    da = det["arousal_smooth"].to_numpy(float)
    if calib is not None:
        cv, ca = calib
        feats = np.vstack([dv, da, np.ones_like(dv)]).T
        dv, da = feats @ cv, feats @ ca
    return dict(
        id=trial_dir.name,
        cat=sess.get("category"),
        dt=dt,
        dv=dv,
        da=da,
        sess=sess,
    )


def attach_command(face: dict, src: pd.DataFrame, smooth_tau: float, clip_command: bool = True) -> dict:
    st = src["timestamp_s"].to_numpy(float)
    v = src["valence"].to_numpy(float)
    a = src["arousal"].to_numpy(float)
    if clip_command:
        # proto_anchor 把部分指令推出了脸可渲染的 [-1,1]；饱和段脸上按构造无法变化，
        # 事件应基于「可渲染指令」定义。
        v, a = np.clip(v, -1, 1), np.clip(a, -1, 1)
    sv = ema_1d(st, v, smooth_tau)
    sa = ema_1d(st, a, smooth_tau)
    out = dict(face)
    out.update(st=st, sv=sv, sa=sa)
    return out


def clustered_direction(df: pd.DataFrame, thr: float, rng: np.random.Generator) -> dict:
    """Trial-clustered direction accuracy: overlapping windows within a trial are
    not independent, so aggregate to per-trial accuracy and bootstrap over trials."""
    big = df[df["d_src"].abs() > thr].copy()
    if len(big) == 0:
        return {"threshold": thr, "n_events": 0}
    big["hit"] = (np.sign(big["d_src"]) == np.sign(big["d_det"])).astype(float)
    per = big.groupby("id")["hit"].mean().to_numpy()
    boots = np.array([rng.choice(per, size=len(per), replace=True).mean() for _ in range(5000)])
    return {
        "threshold": thr,
        "n_events": int(len(big)),
        "n_trials": int(len(per)),
        "pooled_accuracy": float(big["hit"].mean()),
        "trial_mean_accuracy": float(per.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_boot_gt_chance": float(np.mean(boots <= 0.5)),
    }


def analyze_system(
    trials: list[dict],
    window: float,
    n_perm: int,
    seed: int,
    fixed_lag: float | None = None,
) -> dict:
    lag_grid = np.arange(-1.0, 2.51, 0.05)
    for tr in trials:
        # A per-condition free lag lets noise-driven faces (static/random) cherry-
        # pick alignment; use one fixed pipeline lag for a fair comparison.
        lag = fixed_lag if fixed_lag is not None else best_lag(tr, lag_grid)
        tr["lag"] = lag
        tr["DV"] = np.interp(tr["st"] + lag, tr["dt"], tr["dv"], left=np.nan, right=np.nan)
        tr["DA"] = np.interp(tr["st"] + lag, tr["dt"], tr["da"], left=np.nan, right=np.nan)

    dfv = swing_events(trials, "valence", window)
    dfa = swing_events(trials, "arousal", window)
    rng = np.random.default_rng(seed)
    # Surrogate needs enough finite samples per trial; skip short ones
    long_enough = []
    for tr in trials:
        m = np.isfinite(tr["DV"])
        if int(m.sum()) >= 12:
            long_enough.append(tr)
    if long_enough:
        surr_v = surrogate_test(long_enough, "valence", n_perm, rng)
        surr_a = surrogate_test(long_enough, "arousal", n_perm, rng)
    else:
        surr_v = surr_a = {"n_significant": 0, "n_trials": 0, "fisher_combined_p": None}
    return {
        "n_trials": len(trials),
        "window_s": window,
        "mean_lag_s": float(np.mean([tr["lag"] for tr in trials])),
        "event_2afc": {
            "valence": [direction_accuracy(dfv, thr) for thr in (0.3, 0.5, 0.8)],
            "arousal": [direction_accuracy(dfa, thr) for thr in (0.2, 0.3)],
        },
        "event_2afc_clustered": {
            "valence": clustered_direction(dfv, 0.3, rng),
            "arousal": clustered_direction(dfa, 0.2, rng),
        },
        "variance_decomposition": {
            "valence": variance_decomposition(trials, "valence"),
            "arousal": variance_decomposition(trials, "arousal"),
        },
        "surrogate": {"valence": surr_v, "arousal": surr_a},
        "face_dynamics": {
            "mean_V_std": float(np.mean([np.nanstd(tr["dv"]) for tr in trials])),
            "mean_A_std": float(np.mean([np.nanstd(tr["da"]) for tr in trials])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("RAL/experiment/C/C1/result/multicond_noanchor_20260731"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("RAL/experiment/C/C1/output/multicond_noanchor_20260731"),
    )
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument(
        "--fixed-lag",
        type=float,
        default=0.5,
        help="Shared execution+detection lag (s) applied to all conditions; "
        "pass a negative sentinel like -99 to fall back to per-trial best lag.",
    )
    ap.add_argument("--smooth-tau", type=float, default=0.6)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-calibrate", action="store_true", help="不做静止脸检测器校准")
    ap.add_argument("--no-clip-command", action="store_true", help="不把指令裁剪到可渲染 [-1,1]")
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    calib = None if args.no_calibrate else fit_static_calibration(root)
    if calib is not None:
        print(f"[calib] V coefs={np.round(calib[0], 3)}  A coefs={np.round(calib[1], 3)}", flush=True)

    # Shared commanded trajectories from ours
    cmds = {}
    for tid_dir in sorted((root / "ours").iterdir()):
        if not tid_dir.is_dir():
            continue
        src = tid_dir / "source_va.csv"
        if src.is_file() and len(pd.read_csv(src)) >= 5:
            cmds[tid_dir.name] = pd.read_csv(src)

    all_results = {
        "protocol": "commanded=ours/source_va; detected=each condition face VA",
        "smooth_tau_s": args.smooth_tau,
        "systems": {},
    }
    summary_rows = []

    for sys in SYSTEMS:
        trials = []
        for tid, src in cmds.items():
            face = load_face(root / sys / tid, args.smooth_tau, calib)
            if face is None:
                print(f"[skip] {sys}/{tid} no mirror", flush=True)
                continue
            trials.append(attach_command(face, src, args.smooth_tau, not args.no_clip_command))
        print(f"[{sys}] trials={len(trials)}", flush=True)
        fixed_lag = None if args.fixed_lag < -10 else args.fixed_lag
        res = analyze_system(trials, args.window, args.n_perm, args.seed, fixed_lag)
        all_results["systems"][sys] = res

        vc = res["event_2afc_clustered"]["valence"]
        ac = res["event_2afc_clustered"]["arousal"]
        vd = res["variance_decomposition"]["valence"]
        ad = res["variance_decomposition"]["arousal"]
        summary_rows.append(
            {
                "system": sys,
                "n_trials": res["n_trials"],
                "mean_lag_s": round(res["mean_lag_s"], 3),
                "V_dir_acc_thr0.3": vc.get("trial_mean_accuracy"),
                "V_dir_n": vc.get("n_events"),
                "V_dir_p": vc.get("p_boot_gt_chance"),
                "A_dir_acc_thr0.2": ac.get("trial_mean_accuracy"),
                "A_dir_n": ac.get("n_events"),
                "A_dir_p": ac.get("p_boot_gt_chance"),
                "V_R2_turn": vd["R2_turn_level"],
                "V_R2_token": vd["R2_token_level"],
                "V_R2_gain": vd["R2_gain"],
                "A_R2_turn": ad["R2_turn_level"],
                "A_R2_token": ad["R2_token_level"],
                "A_R2_gain": ad["R2_gain"],
                "face_V_std": res["face_dynamics"]["mean_V_std"],
                "face_A_std": res["face_dynamics"]["mean_A_std"],
                "V_surrogate_n_sig": res["surrogate"]["valence"]["n_significant"],
                "V_fisher_p": res["surrogate"]["valence"]["fisher_combined_p"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out / "c1_multicond_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out / "c1_multicond_full.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Markdown
    def _md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if abs(v) < 1e3 else f"{v:.2e}")
                else:
                    cells.append(str(v))
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([head, sep, *body])

    lines = [
        "# 本场 C1 四条件正式对比",
        "",
        "协议：`commanded = ours/source_va`（同一回复轨迹）；`detected = 各条件面部 VA`。",
        f"事件窗 2.0s，source EMA τ=0.6s，**统一固定滞后 {args.fixed_lag}s**（四条件同一对齐，"
        "避免自由搜滞后让静止/乱序脸拟合噪声占便宜）。",
        "方向命中为 **trial 级聚类** 统计：先算每条 trial 的命中率再对 trial 做 bootstrap"
        "（同一 trial 内的重叠窗口不是独立样本）。",
        ("指令裁剪到可渲染 [-1,1]（proto_anchor 将约 12% token 推出量程，饱和段按构造不可检验）。"
         if not args.no_clip_command else "指令未裁剪。"),
        ("Mirror 读数经**静止脸校准**：用 mishra/xpress 40 段已知显示 VA 的视频拟合线性校准"
         "（检测效价被压缩约 4x 且混入唤醒分量）；校准集与被测数据独立，四条件同一映射。"
         if calib is not None else "未做检测器校准。"),
        "",
        "## 主表",
        "",
        _md_table(summary),
        "",
        "## 读法",
        "",
        "- **方向命中 (V_dir_acc_thr0.3)**：指令轨迹大幅摆动时，脸上是否同向（机会水平 50%）；"
        "p 为 trial 聚类 bootstrap 单侧 p（>0.5）。",
        "- **R² gain**：相对「整句一个表情」，token 级轨迹多解释多少脸上方差（对固定脸基线增益应接近 0）。",
        "- Mishra/Xpress 为话轮级固定表情；random 为打乱时间结构；ours 为逐 token。",
        "",
    ]
    (out / "c1_multicond_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {out / 'c1_multicond_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
