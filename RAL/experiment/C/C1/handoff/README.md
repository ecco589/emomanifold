# 实机交接说明（C1+C2 重跑）

本目录仅保留**带到工控机的文档与环境模板**。脚本与刺激句在仓库固定位置，勿在此复制第二份。

| 需要 | 路径 |
|------|------|
| 20 条刺激句 | `../C1/data/stimuli.csv` |
| Mishra / Xpress 脚本 | `../code/baselines/` |
| OpenAI 密钥模板 | `env/openai.env.example` |
| 实机 Cursor 说明 | `给实机Cursor.txt` |
| 传入传回约定 | `传入传回约定.txt` |
| Windows Mirror 批处理提示 | `WINDOWS_CURSOR_PROMPT_SHORT.txt` |

**录完传回后落盘位置：**

- 四条件录像 + Mirror → `../result/multicond_YYYYMMDD/`
- 分析表 → `../output/multicond_YYYYMMDD/`（跑 `../code/run_c1_multicond_compare.py`）
- 实机延迟 → `../../D/out_robot/`

**不要带进工控机的：** 大模型权重、`face/` 舵机包（实机旧环境应已有）、本机分析产物 zip。
