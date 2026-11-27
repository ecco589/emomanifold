# 细节修改建议（latex-paper-en 诊断 + 人工核对）

对象：`RAL/emotion_manifold_en.tex`（行号按 2026-10-28 版本）。
只保留核对原文后仍成立的问题；脚本误报已剔除，见文末“不用改”。

## 一、必须改

### 1. 人类偏好的两个数字互相矛盾（约 L285）
- 现状：选择比例 EmoManifold 83.5%、打乱 11.5%、静态 5.0%，括号里又写 “all 20 participants preferred EmoManifold”。
- 问题：按试次有 16.5% 没选 EmoManifold，读者会以为“20 人每次都选”与 83.5% 冲突；Wilcoxon 的单位（试次还是被试）也不明。
- 改法：写清单位。例如 “83.5% of trials; every participant chose EmoManifold in the majority of their trials (one-sided Wilcoxon over participants, $p<10^{-6}$)”。以实际统计口径为准。

### 2. 摘要 ΔV≈−0.22 对不上正文主数字（L37 vs L213/L215）
- 正文：句级 ΔV(S−P) = −0.19；锚定正性关键词处低 0.22。
- 摘要把 “word-level” 和 “cross-context valence gap ΔV≈−0.22” 放在一起，读者对着 Fig. 4(a) 找不到 −0.22。
- 改法：摘要明确指关键词级，或换成直观指标（见 `suggest_general.md` Step 2 的指标表）。

### 3. R² 口径不统一（L37 / L256 / Fig. 6 图注）
- 摘要 “22 and 19 percentage points”；正文 “additional 21.8%/18.9% of facial variance”；图注是 percentage points。
- 改法：全文统一为 percentage points（pp），正文不要用 “%”，以免读成相对提升。

### 4. 延迟表一格两套系统（L258–L271）
- `10.2±4.3 / 9.9±4.0` 的斜杠左右分别是 GPT-4o-mini 与 Qwen2.5-0.5B，不看正文看不懂。
- 改法：拆成两列（TTFF, GPT-4o-mini｜TTFF, Qwen2.5-0.5B），或在表头写明。
- 补一句原因：同为本地 0.5B，Xpress 26.5 s、Mishra 9.9 s，差一倍多；现有“要等整句”的解释不覆盖这个差距（可能是 Xpress 输出更长 / 两段式提示，以实际为准）。
- `\resizebox` 会把表字号缩到正文以下，能不用就不用。

### 5. 相机标定只用基线的静态片段（L252）
- 40 段标定来自 Mishra/Xpress 整句静态表情，冻结后评估 EmoManifold 的动态混合表情；87.5% / 69.6% 都建立在这个线性映射上。
- 审稿人会问映射在动态、混合表情上是否有效。补一句适用理由，或补一个在动态片段上的验证。

## 二、其次

### 6. 五个还是六个模型（L163 vs Fig. 2 / Table I）
- 正文说词表交集来自 “five comparison models (2B–8B)”，图表是六个，多出来的是 Qwen2.5-0.5B（代码 `RAL/experiment/A/A2/code/run_e2.py` 也是 5 + 1）。
- 改法：写明交集按五个外部模型的分词器取，0.5B 在同一张词表上另测。

### 7. 相关工作一整段，后半与引言重复（L67–L69）
- 第二段先把“回合级标签”又讲一遍。拆段，并把 “training-free、直接用 unembedding 矩阵” 留作相对 interpretability 工作的缺口。
- 结构调整见 `suggest_general.md` Step 3。

### 8. 结论没收回引言三条贡献（L56–L61 vs L299–L302）
- 0.7 ms、两模型反讽测试、29-DoF 四条件评测在结论里都没点名。各用半句收束，不加新结果。

### 9. 图格式
- `fig1.png`、`fig3.png` 若还能编辑，导出 PDF（矢量）；照片类 `fig_add.png`、`fig7.png` 保持位图即可，确认 ≥300 dpi。
- 注：源文件名有 fig5 → fig7 的跳号，但编译后的图号是连续的（Fig. 1–7），不影响审稿。

### 10. 模型名
- Table I 的 “Gemma-2.2-2B”：公开检查点通常写作 Gemma-2-2B，投稿前核对实际使用的模型名。

### 11. RA-L 格式
- 摘要上限 200 词（RA-L 规定，不是 IEEE 通用的 250）；当前约 148 词。
- 终稿在摘要下放 2–5 个关键词（`IEEEkeywords`），前两个须来自 RA-L 词表；初稿在 PaperCept 里填。
- 初稿按双盲：作者信息留位置但不写实名（当前是占位符，保持）。

### 12. 参考文献
- 22 条全部被引用，无缺失条目。
- 三条是 arXiv 却用 `@article`：`li2025x2c`、`lee2023safe`、`qwen2024qwen25`。查是否已正式发表；未发表的改 `@misc` 并写 arXiv 号。
- 其余条目缺 DOI（IEEE 建议项，非错误），建议补齐。
- 本次没有把题录发到外部数据库核对；正式投稿前建议逐条核对。

## 三、不用改（脚本误报）

- 摘要“缺结论句”：最后一句已经是意义句。
- 摘要 148 词低于 150：RA-L 只有 200 上限。
- 标题里 “LLM” 缩写：无需展开。
- 把摘要问题句挪到方法句后、删 L256 的 “merely”、把 “Fig. X shows” 改过去时、把对比句一律当 AI 腔删除：都会伤论证或违背 IEEE 时态习惯。
- 破折号约 18 处：只在改 L69 那种三重插入语时顺手拆开。

## 附：诊断过程

- Skill：`.agents/skills/latex-paper-en`（v6.0.0）。
- 脚本需 Python ≥3.10（用 `/opt/homebrew/bin/python3.12` 运行；系统 3.9 会报错）。
- 已跑：bibliography、figures、tables、abstract、grammar、sentences、deai、claim-forward、logic（含 cross-section / motivation-thread）、literature、experiment、title。
- 未跑全：format（本机无 chktex）、figures 的 DPI 检查（无 Pillow）。
