# Mishra / Xpress 话轮级 SOTA 标签（C1 对照用）

对 **同一段回复文本** 做整句情绪分类，供 C1 四条件实机对照中的 mishra_gpt / xpress_gpt 条件使用。

| 文件 | 说明 |
|------|------|
| `run_mishra.py` | Mishra 式 1–6 类分类（GPT / 本地 Qwen） |
| `run_xpress_conv.py` | Xpress 式 emotion+intensity（GPT / Qwen） |
| `weights_from_label.py` | 标签 CSV → 7 维 FKCN 权重 |
| `mishra_prompt.txt` / `xpress_conv_prompt.txt` | 提示词 |
| `xpress_label_map.json` | Xpress 标签 → 7 类映射 |
| `third_party/Xpress/` | 官方提示参考片段 |

预跑结果（开发机）→ `../../result/label_precompute/`。

```bash
python RAL/experiment/C/C1/code/baselines/run_mishra.py --model gpt
python RAL/experiment/C/C1/code/baselines/run_xpress_conv.py --model gpt
```

本文逐 token 权重导出见 `../../../D/code/export_ours_weights.py` → `D/out/precompute/`。
