# EmotionManifold → RAL 投稿大纲

**标题暂定**：*Token-Level Continuous Affect from LLMs for Real-Time Robot Facial Expression*

**定位**：以**客观指标**为主，强调逐 token 连续情感的**提取精度、时序同步性、执行保真度**；不做主观偏好实验，不声明"更自然"等人机交互常见的主观化措辞。

---

## Abstract（~150 词）

- **问题**：对话机器人表情多为句级/离散标签，与语言细粒度不同步，无法表达句内情感漂移（讽刺、否定、转折）
- **方法**：LLM 流式生成中逐 token 投影 VAD → FKCN 融合 7 基元表情 → 29 DOF 实机
- **亮点**：免微调、端到端实时、句内情感可跟随、客观可验证
- **结果**：1 句精度/同步性指标 + 1 句实机保真度结论

---

## I. Introduction

- 动机：表情与话语**子句级不同步**，离散/句级方法结构上无法跟随句内情感漂移
- **贡献**：
  1. 提出一种**免训练、模型无关**的 token 级连续情感读回方法：仅由 LLM 解嵌入矩阵的情感词簇构造 VAD 轴，无需微调或外部标注，可挂载于任意开源 LLM 并**实时本地推理**；
  2. 在公开情感数据集上**递进验证**前向投影能力：句级 VAD 与人工/权威映射标签对齐，并在讽刺等复杂语境中对关键情绪词做 token 级对照，证明读回是上下文敏感的；
  3. 提出一套泛化性极强的部署管线：面部机器人自由度高、逐 DOF 调试极为繁琐，而本管线由 FKCN 将连续 VAD 映射为n种基础表情权重(本工作n为7)，再经面向执行的校准与融合下发到具体 DOF，因此本方法可轻松迁移到其他面部机器人平台；
  4. 在 29-DOF 实机上提出 **镜像测试（Mirror Test）** 等客观实验验证管线：对实机进行摄像，通过面部表情识别（真人数据集训练）情绪，对原始VAD值进行对比，以及人工打分，证明本系统具有高保真度。

---

## II. Related Work（精简 ~0.5 页）

暂略

## III. Method

System Architecture（核心图）以及总体概述

### A. Emotion Axis Mining
- 解嵌入矩阵 + 情感词簇 → Valence / Arousal（/ Dominance）轴
- AdaptiveScaler 标定（正文一笔带过）

### B. Token-Level Streaming Projection
- 流式 hidden state 投影；句级聚合（末 token / 加权平均）用于与句级标注对比

### C. FKCN: VAD → 7-Way Expression Weights
- 模糊 Kohonen 原型 → 7 基元权重融合（含 VAD 流形几何解释，1 图）

### D. Actuator Mapping: 7 Expressions → 29 DOF
- YAML factor 融合，约束 [0,1]；10 fps 帧插值

```
Text → LLM stream → VAD(t) → FKCN → 29-DOF face
```

---

## IV. Experiments

### A. 情感子空间可靠性

#### E1（已完成）：极词数 sweep × Spearman

- **设置**：Qwen2.5-0.5B；英文 pole 词 N=10–80；NIPS 五模型交集 hold-out（300 词）；极词与 hold-out **不互斥**（对齐 NIPS B1）。
- **图**：六模型（NIPS 五模型 + Qwen 0.5B）Spearman **均值实线 + min–max 带**（V/A/D 三栏），见 `RAL/experiment/E1/output/e1_spearman_pole_sweep.pdf`。
- **结论要点**：修正极词筛选后，0.5B 在 V 维可进入六模型带；A/D 仍偏弱，支持「小模型 + 轴质量」而非流程错误。

#### E2（已完成）：跨模型 Spearman（NIPS Table 1 + Qwen2.5-0.5B）

- **设置**：固定 pole_n=50；英文 NRC-VAD hold-out Spearman（ρ_V, ρ_A, ρ_D）。
- **数据**：NIPS 五模型来自 `E1/data/nips_b1_per_model_spearman.json`；Qwen2.5-0.5B 来自 `E1/result/en/pole_50`。
- **表**：`RAL/experiment/E2/output/e2_table.tex`（复现 NIPS Table 1 并加入本模型一行）。
- **结论要点**：0.5B 与 NIPS 五模型同量级，支持方法跨 backbone 泛化。

#### E3（已完成）：EN vs ZH 极词 × 轴正交 + 跨语聚类

- **两条件**（每极 $N{=}50$，Qwen2.5-0.5B）：EN-only（E1 `en/pole_50`）、ZH-only（E1 `zh/pole_50`）。
- **指标**：$|\cos(V,A)|, |\cos(A,D)|, |\cos(V,D)|$；5 簇跨语 mean cosine vs **共享**随机 3D 基线（0.410）。
- **表**：`RAL/experiment/E3/output/e3_table.tex`。

### B. 投影能力（递进验证）

**设计原则**：先用**句级 VAD 金标准**证明整体映射合理，再用**同词跨语境 token 对照**证明复杂情感（讽刺等）的上下文敏感性。两层共用同一前向管线，指标随层级递进。

#### 标签 → VAD 的权威参照（Ground Truth 构造）

| 数据源 | 标签类型 | VAD 参照 | 依据 |
|---|---|---|---|
| **EmoBank** (~10k 句) | 原生 VAD (V/A/D, 1–5) | 直接用 reader 视角标注，线性归一化至与投影同尺度 | Bostan et al. 2017；Russell & Mehrabian 1977 |
| **GoEmotions** (~58k 句, 27 类) | 细粒度离散情感 | 各类别 → NRC-VAD 词表质心（或 Büchel et al. 2016 BE↔VAD 回归映射） | Mohammad 2018；Demszky et al. 2020 |
| **iSarcasmEval** (Subtask C) | 讽刺 vs 非讽刺改写对 | 改写句（rephrase）的句级 VAD 作"真实意图"；讽刺句表面 VAD 作对照 | Abu Farha et al. 2022；Leggitt & Gibbs 2000 |

> EmoBank 为**主评测**（原生 VAD，无映射误差）；GoEmotions 验证跨 taxonomy 泛化；iSarcasmEval 专测复杂情感。

#### B1. 句级投影对齐（整体合理性）

- **数据**：EmoBank reader VAD（主）；GoEmotions 高置信子集（辅，≥2 rater 一致）
- **流程**：句子前向传播 → 句级聚合 VAD → 与 GT 对比

#### B2. 复杂情感：关键情绪词 token 级对照（上下文敏感性）

- **数据**（按优先级）：
   **iSarcasmEval Subtask C**：讽刺句 vs 同义非讽刺改写，天然最小对立对
- **方法**：
  1. 定位**关键情绪词/讽刺触发词**
  2. **对照 A**：同词在 iSarcasmEval 非讽刺改写句 / 其他中性句中的 VAD（词典先验或简单句投影）
  3. **对照 B**：讽刺句 vs 改写句的**句级 VAD 差向量**（意图方向）

### C. 实机实验（RAL 重点，三层客观证据，无主观偏好）

1. **镜像测试 Mirror Test（主打，带独立参照）**
   `文本 VAD → 实机 → 摄像头 → FER/OpenFace AU（或 VLM 裁判）→ 还原 VAD → 与原始 VAD 相关`
   全自动 round-trip：检验「情感指令是否如实映现在脸上」

2. **人类识别准确率（客观，非主观偏好；低成本 N≈15–20）**
   仅看机器人脸（去文字/语音）→ Geneva Emotion Wheel 选情感 / VA 平面标点 → 与意图 VAD 算**准确率/误差**；

---

## V. Discussion & Limitations

- 10 fps 与舵机物理极限的权衡
- 未做主观接受度评估（scope 之外，列为 future work）
- 模型/语言/文化泛化

---

## VI. Conclusion
暂略
