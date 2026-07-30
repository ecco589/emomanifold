# D 部署延迟

| 目录 | 来源 | 说明 |
|------|------|------|
| `out/` | 开发机预跑 | `run_latency_d.py` |
| `out/precompute/` | ours 逐 token 权重预跑 | `export_ours_weights.py` |
| `out_robot_noanchor_20260731/` | **实机工控机** | 2026-07-31 与 C1 主数据同场同语料；**论文用此表** |

实机口径说明见 `out_robot_noanchor_20260731/LATENCY.md`。
旧场延迟（20260730，数字量级一致）已归档至 `RAL/experiment/_archive_obsolete_20260731/D/out_robot/`。

```bash
python RAL/experiment/D/code/run_latency_d.py   # 开发机复跑
```
