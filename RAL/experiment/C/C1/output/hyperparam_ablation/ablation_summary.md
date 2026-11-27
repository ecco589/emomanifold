# C1 offline hyperparameter ablation (α × prototype set)

- Field: `RAL/experiment/C/C1/result/multicond_noanchor_20260731/ours`  |  trials: 20
- Window: 2.0 s; event thresholds V>|0.3|, A>|0.2|
- Pipeline: clip(command VA) → FKCN → EMA(α) → 10 fps smoothstep + 0.05/frame rate limit → VA readout on fixed P=13 centroids

## Scope

This replays the **software** stage only; there is no video and no Py-Feat. The hardware stage (servos → face → camera → detector) is common-mode across conditions because α and P do not enter it, and it was measured once at the default setting in the main experiment. Compare the default row below against the camera numbers (V 87.5%, A 69.6%) to judge how much the missing stage costs.

## Deviations from the Fig. 5A protocol

1. Rate limit runs in 7-D weight space (29-D presets live on the robot PC); weight-space deltas upper-bound factor deltas, so the limiter binds earlier here than on hardware (conservative).
2. VA readout always uses P=13 centroids, so prototype sets share one coordinate system.
3. Events use a single fixed lag of 0.0 s for all conditions with no per-trial fitting. The 0.5 s of Fig. 5A is a hardware pipeline delay that does not exist offline; the smoothing lag column is a diagnostic, not an alignment parameter.

## Results

| α | set | P | V dir-acc | V floor | A dir-acc | A floor | V amp-ret | lag (s) | sat | frame |Δw| |
|---|-----|---|-----------|---------|-----------|---------|-----------|---------|-----|------------|
| 0.3 | 13 | 13 | 89.9% (n=129, p=3.89e-22) | 47.7% | 72.5% (n=153, p=1.12e-08) | 40.5% | 0.794 | 0.39 | 4.7% | 0.0104 |
| 0.6 **(default)** | 13 | 13 | 87.6% (n=129, p=1.82e-19) | 45.3% | 70.6% (n=153, p=1.86e-07) | 40.0% | 0.854 | 0.23 | 10.5% | 0.0166 |
| 0.9 | 13 | 13 | 81.4% (n=129, p=1.4e-13) | 46.4% | 71.2% (n=153, p=7.53e-08) | 44.1% | 0.820 | 0.32 | 16.0% | 0.0214 |
| 0.3 | 7a | 7 | 89.9% (n=129, p=3.89e-22) | 47.2% | 77.8% (n=153, p=1.48e-12) | 42.3% | 0.782 | 0.32 | 4.8% | 0.0107 |
| 0.6 | 7a | 7 | 85.3% (n=129, p=4.55e-17) | 44.9% | 73.9% (n=153, p=1.47e-09) | 43.7% | 0.841 | 0.23 | 11.5% | 0.0170 |
| 0.9 | 7a | 7 | 80.6% (n=129, p=5.98e-13) | 45.5% | 71.2% (n=153, p=7.53e-08) | 45.2% | 0.812 | 0.25 | 16.9% | 0.0217 |
| 0.3 | 7b | 7 | 90.7% (n=129, p=4.27e-23) | 46.4% | 74.5% (n=153, p=5.08e-10) | 41.7% | 0.820 | 0.38 | 5.2% | 0.0106 |
| 0.6 | 7b | 7 | 86.8% (n=129, p=1.22e-18) | 43.8% | 73.9% (n=153, p=1.47e-09) | 42.3% | 0.876 | 0.27 | 11.2% | 0.0169 |
| 0.9 | 7b | 7 | 83.7% (n=129, p=1.33e-15) | 46.0% | 73.2% (n=153, p=4.13e-09) | 45.0% | 0.849 | 0.29 | 17.1% | 0.0216 |

`V floor` / `A floor`: same chain on time-shuffled commands (5 permutations), scored against the true commands. `sat`: share of per-dimension frame steps clamped by the 0.05 rate limiter.

## Paired comparison against the default condition

Trial-level accuracies are paired across conditions (same 20 utterances), so differences are tested pairwise rather than by comparing marginal intervals.

| α | set | ΔV vs default | ΔA vs default |
|---|-----|---------------|---------------|
| 0.3 | 13 | +2.1 pp [+0.0, +4.8], p=0.109 | +4.2 pp [-2.2, +11.2], p=0.260 |
| 0.6 | 13 | — (default) | — (default) |
| 0.9 | 13 | -9.0 pp [-21.2, -1.2], p=0.042 | +2.5 pp [-4.0, +9.8], p=0.498 |
| 0.3 | 7a | +2.1 pp [+0.0, +4.8], p=0.109 | +6.5 pp [+0.9, +12.8], p=0.069 |
| 0.6 | 7a | -2.3 pp [-5.2, +0.0], p=0.102 | +2.4 pp [-0.3, +5.3], p=0.173 |
| 0.9 | 7a | -5.0 pp [-9.0, -1.2], p=0.043 | +1.9 pp [-3.9, +8.7], p=0.461 |
| 0.3 | 7b | +2.3 pp [+0.0, +5.1], p=0.109 | +5.2 pp [-0.7, +11.7], p=0.183 |
| 0.6 | 7b | -2.8 pp [-8.3, +0.0], p=0.317 | +3.2 pp [-0.5, +7.8], p=0.225 |
| 0.9 | 7b | -7.0 pp [-19.6, +1.5], p=0.249 | +4.5 pp [-1.1, +11.2], p=0.136 |

## Suggested IV-C sentences (draft)

Note: codebase EMA uses α·cur+(1-α)·prev; smaller α damps more.

离线重放给出的效价方向一致率落在 75%--87%（唤醒 68%--74%），而同一管线在打乱时序的指令上仅 44%--48%，说明该指标并非恒过。默认设定（α=0.6、P=13）为 84.4%/67.7%，与相机实测 87.5%/69.6% 量级一致。逐试次配对比较显示：α=0.9 较默认低 9.0 个百分点（95% CI [-21.2, -1.2]，Wilcoxon p=0.042），此时 16% 的逐帧步长被单帧上限截断（默认 11%），限速成为瓶颈；α=0.3 与默认无显著差异（+2.1 pp，p=0.109）但平滑滞后由 0.23 s 升至 0.39 s。两种 P=7 构造（保留每类中的不同原型）与默认亦无显著差异（-2.3 pp / -2.8 pp）。
