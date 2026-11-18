# RAL 实验：投稿前补充计划（版面受限版）

目录地图见 [`STRUCTURE.md`](STRUCTURE.md)。

**约束：** RAL 正文最多 8 页，无版面新增图/表；每项补充只落「某表加一行」「某图加一根柱」或「正文一句话」。  
**投稿附带：** project page 脚注 URL（可选）。  
**人因伦理（不做 IRB）：** IV-C 写知情同意 + 匿名；Discussion 写「小样本匿名偏好，完整伦理备案与更大样本复验留待后续」。不写假批件。

**补充 backbone：** `Llama-3.2-3B`（与表 I / NIPS 五模型中的 `llama32_3b` 同一模型；**在 Modal 上跑**）。  
**部署 backbone（C/D）：** 仍为实机 `Qwen2.5-0.5B`，不再换模型重录。

---

## 版面腾挪（先做）

| 动作 | 原因 |
|------|------|
| 图 3 | **暂保留**（作者决定先不删）；表 II 已扩为 Qwen+Llama 四行 |

---

## 可复用资产（Modal 前先核对）

**NIPS 检出目录（本机已有）：** `/Users/eccoliu/EmotionManifold-NIPS`  
（对应 git 分支/标签 `finalV1` = `FinalNIPS`；Modal 脚本在 `modal_experiments/b1_reform_llama32.py`）

| 资产 | 路径 / 状态 | 能否直接用 |
|------|-------------|------------|
| Llama Spearman（A1/A2） | 表 I 已有；NIPS `results/B1/llama32_3b/pole_50/report.json` 内 ρ 与表 I 一致 | ✅ **A1/A2 不必重做** |
| **Llama EN 极词 + scaler（N=50）** | `EmotionManifold-NIPS/results/B1/llama32_3b/pole_50/report.json` → 字段 `word_groups`（六极各 50 词）+ `scaler`（V/A/D 的 scale/bias） | ✅ **直接复用**；不必重拟合 scaler。轴向量未单独存盘，但 `axes_identical: true` → Modal 上加载 Llama Wout + 这批极词即可确定性重建 |
| Llama EN 其它 N | 同目录 `pole_{10..80}/`（部分档另有独立 `scaler.json` / `lexicon.json`） | 备用；正文默认 **pole_50** |
| Llama ZH 极词 / scaler | NIPS `results/` 下**未找到** llama32 中文构轴产物 | ❌ A3 若要「Llama × ZH」一行，仍须 Modal 补跑中文极词构轴 + 拟合 scaler |
| Qwen EN/ZH（现表 II） | `RAL/experiment/A/A1/result/{en,zh}/pole_50/` | ✅ 不动 |
| B1 三元组 | `RAL/material/b1_sarcasm_triplets/` | ✅ |

**已导出 EN 资产：** `RAL/experiment/A/A3/data/llama32_3b/en/pole_50/`（`pole_lexicon.json` / `scaler.json` / `report.json`）。

**Modal 脚本（L4）：** `RAL/experiment/modal/ral_llama32_a3_b1.py`  
- EN：复用上列资产重建轴 → A3 EN + B1  
- ZH：PRIORITY_ZH + NRC 极值候选（top_k=300）→ **Llama 分词器单 token 过滤** → 构轴 + 拟合 scaler → A3 ZH  

```bash
# 若 modal secret list 报 Token not found：
modal token new
# 需已有 Modal secret: huggingface（含 HF_TOKEN，Llama gated）

cd /Users/eccoliu/Desktop/EmotionManifold
python -m modal run RAL/experiment/modal/ral_llama32_a3_b1.py
```

---

## 按实验编号：补什么、怎么写进论文

### A1（极词数扫描，图 2）— **不做**

- Llama-3.2-3B 已在 NIPS 五模型 sweep / 图 2 范围带内；无需加模型、无需重跑。

### A2（跨模型 Spearman，表 I）— **不做**

- 表 I 已有 `Llama-3.2-3B` 一行（ρ_V=0.800, ρ_A=0.828, ρ_D=0.708）。

### A3（轴正交 + 跨语簇，表 II）— **Modal**

- **补：** Llama-3.2-3B：EN 直接用 NIPS `pole_50/report.json` 的极词+scaler 重建轴后算正交/凝聚度；**ZH 在 NIPS 中无现成产物**，要四行（Qwen×2 + Llama×2）则 Modal 补跑中文构轴，否则可只加 Llama EN 一行。
- **写法：** 表 II 最左加一列 **Model**；上两行 Qwen（EN / ZH），下接 Llama（EN，及可选 ZH）。caption 写明 backbone × 构轴语言。
- **成本：** EN 几乎只剩加载模型 + 前向几何；ZH 才需要重新构轴/拟合 scaler。

### B1（讽刺三元组，图 4）— **Modal**

- **补 1：** Llama-3.2-3B + 上一步落盘的 EN 轴/scaler，重跑 200 组三元组（纯前向）。
- **补 2（高优先级，本地即可）：** GPT-4o-mini 对同一批 S/P 整句打离散情绪标签，统计 S 与 P 被判为同一类的比例（不依赖 Llama）。
- **写法：** 图不动；正文两句：
  - Llama 上关键词级效价差 −0.XX，与 Qwen 趋势一致。
  - 离散标签基线：S/P 同引语下 XX% 被判为同一情绪类 → 话轮级标签在讽刺上失效。

### 单 backbone 实机（C）怎么写 — **放 IV-C 开头，一句**

**审稿人建议：放在 IV-C / C 段开头，不要放 Discussion，也不要放 B 结尾。**

- B 结尾写「所以实机只用一个模型」会把语境实验和硬件选择缠在一起。  
- Discussion 写像在道歉，且易被当成局限主句。  
- C 开头写是自然分工：表征已跨模型验证，闭环评测绑定部署机。

**建议句（可改）：**  
「闭环实机与人因评测使用端侧部署的 Qwen2.5-0.5B；第 IV-A/B 节已表明同一构轴流程在 Llama-3.2-3B（表 I 已含）上具有可比的轴几何与语境敏感性，故不在第二 backbone 上重复相机闭环。」

### C1（实机四条件 + 稳健性，图 5 / 图 6，**IV-C**）

**主实验（已完成）：** 沿用现有四条件（ours / Mishra / Xpress / shuffled）及 20260731 无锚定场；图 5 不动；**不换 Llama 重录**。

**IV-C 结构：**
1. **开头一句：** 单 backbone 理由（见上）。
2. **主结果：** 现有图 5 / 表 III / 人因叙述。
3. **末尾同一段「稳健性与跨载体复现」**（不进 C2、不在 Discussion 报数字）：

#### 超参消融（离线，无新表）✅

- **已跑：** `C1/code/run_c1_hyperparam_ablation.py` → `C1/result/hyperparam_ablation/ablation_summary.json`（+ `output/.../ablation_summary.md`）
- **⚠️ 它测的不是 C1 那件事。** C1 主实验测 `指令 →[A 软件]→[B 物理: 舵机→脸→相机→Py-Feat]→ 回读`；离线消融**没有视频**，只测 A 段输出（帧率权重轨迹按质心折算回 VA）。
- **为什么仍不需要实机：** α/P **只存在于 A 段**；B 段（同一机器人/预置向量/机位/检测器/校准）对六组超参共模，且已在默认 α=0.6 下实测过。于是「A 段还剩多少转折结构」离线算，「B 段吃掉多少」用主实验那一点定标。
- **定标结果（关键）：** 默认 α=0.6、P=13 离线 V=84.4%、A=67.7%，相机实测 87.5%/69.6% → B 段在 2 s 窗方向指标上几乎透明，故扫 A 段成立。
- **论证性质：** 这是**基于分解的推断**，不是六组各自的测量；假设 B 段对 A 段输出的性质不敏感。**α=0.9 最可能偏离**（最抖，舵机机械低通 + 100 ms 移动时间对抖动信号的衰减可能不同）。真要补，只补 α=0.9 的 2–3 条。
- **离线管线：** `multicond_noanchor_20260731/ours/*/source_va.csv` → clip → FKCN(P) → EMA(α) → `FaceFrameInterpolator(10fps, 150ms, 0.05)` 重放 → 帧率轨迹按**固定 P=13 质心**读回 VA → C1 同款 2 s 窗与阈值。
- **三处已记录的偏差：** ① 29 维预置向量在实机 `face/face_driver.py`，故插值/限速在 7 维权重空间执行（权重增量是因子增量上界 → 限速更早触发，属保守）；② 读回质心固定 P=13，避免 P 比较的循环性；③ **对齐滞后不同**——离线无执行-检测滞后，故不套用图 5A 的固定 0.5 s。
- **⚠️ 滞后的坑（已修）：** 曾用「逐试次估计滞后」对齐，等于给每条试次一个自由参数，而这正是 `run_c1_multicond_compare.py` 明确拒绝的做法（自由滞后会让噪声条件挑选对齐）。现改为**六组统一零滞后、不作拟合**（`--event-lag-s`）。对照：硬套 0.5 s 会塌到 48–58%（离线无该滞后，属过度平移）；零滞后与估计滞后结果接近，且无自由参数。平滑滞后仅作诊断量，不参与对齐。
- **设计（9 组 = α{0.3,0.6,0.9} × 原型集{13, 7a, 7b}）：** `7a`/`7b` 各保留每类原型中的不同一个 → P=7 结论不依赖"留哪一个"。
- **地板对照：** 同管线作用于**时序打乱**的指令（5 次置换），V **44–48%**、A **40–45%** → 指标会失败，不是恒过。
- **结果（试次级均值，零滞后）：** V **75–87%**、A **68–74%**，九组均远高于地板。
- **配对检验（同 20 条语料，逐试次配对）：** α=0.9 在三种原型集上一致低于默认 **5.0–9.0 pp**（两种 p<0.05；P=13 上 −9.0 pp、CI [−21.2,−1.2]、p=0.042）；α=0.3 无显著差异（+2.1 pp, p=0.109）；α=0.6 下 7a/7b 无显著差异（−2.3 / −2.8 pp, p≥0.10）。唤醒各组均无显著差异。**注意必须用配对**——各组独立 CI 大幅重叠，照那个报法差异不显著。
- **限速饱和率（"限速成瓶颈"的直接证据）：** 被单帧 0.05 上限截断的逐帧步长比例 α=0.3/0.6/0.9 = **4.7% / 10.5% / 16.0%**；对应平滑滞后 0.39 / 0.23 / 0.32 s——α=0.9 抖动最大却**未换来更低滞后**。
- **写法（IV-C）：** ✅ 已写入「稳健性与跨载体复现」段。
- **Discussion：** ✅ 改为「α 与原型数 P 的敏感度见第 IV-C 节」；残余相关句保留。

#### Grimace 跨载体定量（离线回放）❌ 已砍

- **冒烟（013 ours + shuffled）：** 脸检 `valid_ratio=1.0`，但 Py-Feat VA **无语义**（ours 效价相关 ≈0.04，转折同向 ≈57%≈机会水平；ours/shuffled 读出曲线几乎同形）。归档：`C1/result/grimace_smoke/`，对照图 `C1/output/grimace_smoke/smoke_va_overlay.png`。
- **决定：** 不做全量 20×2；**不写**「效价 XX%、唤醒 XX%」。图 6 **示意保留**；IV-C 仅保留「同一权重可驱动双载体」一句，不报数字。
- **渲染脚本（若以后换检测器可复用）：** `C1/code/render_grimace_from_source_va.py`（应用 C1 `source_va`，勿用 `D/ours_weights`）。

### C2（人因问卷）

- **现状：** n=20 已写入正文；扩到 n=40 **非必须**（可不做）。
- **写法：** 知情同意 + 匿名（已定）；不申请 IRB、不写审批号。
- **不含：** 超参消融、Grimace 定量、第二 backbone（见上）。

### D（延迟，表 III）

- **补：** 无。沿用现有五条件延迟表。

---

## 执行顺序

```
✅ EN 极词/scaler 已导出 → A3/data/llama32_3b/en/pole_50/
✅ Modal L4：A3 EN+ZH + B1 已跑通并写入 tex（表 II / IV-B / IV-C 开头）
    ↓
（并行本地）B1 补 2：GPT-4o-mini 离散标签一句（可选）
    ↓
✅ 超参消融离线跑完并写入 IV-C / Discussion
    ↓
❌ Grimace 定量已砍（冒烟：有脸无语义）；图 6 示意保留
    ↓
tex 收尾：人因知情同意 + Discussion 伦理一句（✅）；project page（待）
```

**砍序执行情况：** C2 扩样不做（n=20）→ Grimace 定量已砍（图 6 示意保留）。**不可砍：** B1 离散标签一句（若做）、IV-C 超参消融（✅）、人因知情同意表述（✅）。

---

## 净版面账

| 变动 | 形式 |
|------|------|
| 图 3 | 暂保留 |
| 表 I / 图 2 | **不动**（Llama 已在内） |
| 表 II | ✅ Model 列 + Llama × EN/ZH |
| 正文 IV-C 开头 | ✅ 单 backbone 理由 |
| 正文 IV-B | ✅ Llama 语境一句 |
| 人因 / Discussion | ✅ 知情同意 + 小样本伦理一句（无 IRB） |
| 正文 IV-C 末段 | ✅ 超参消融已写入；Grimace 定量数字不写（示意一句） |
| 仍紧则 | 图 2 三子图合并为单图（三色线）再省几行 |

---

# 附录：历史实验对照与补充记录（已完成步骤）

## 第一步：搭好目录和刺激句 ✅

1. 已建文件夹：
   - `RAL/experiment/C/C1/code/baselines/`（Mishra/Xpress；含 `third_party/`）
   - `RAL/experiment/C/C3/`（含 `叙述.txt`）
   - `RAL/experiment/D/`（含 `out/`、`out_robot/`、`out/precompute/`）
2. 刺激句固定用：`RAL/experiment/C/C1/data/stimuli.csv`（20 句，后面全部共用）。

---

## 第二步：实现 Mishra 对照（无官方代码，按论文提示词写）✅

1. ✅ 已建 `C1/code/baselines/mishra_prompt.txt`（论文分类提示，只回答 1–6）。  
2. ✅ 已写 `C1/code/baselines/run_mishra.py`（解析类别 + 记耗时；支持 `OPENAI_BASE_URL`；Qwen 用 1–6 约束下一 token）。  
3. ✅ 已跑两遍并保存：
   - ✅ GPT → `C1/result/label_precompute/mishra_gpt.csv` + `mishra_gpt_weights/`
   - ✅ 本地 Qwen → `C1/result/label_precompute/mishra_qwen.csv` + `mishra_qwen_weights/`  
     - **注意**：0.5B base 在该提示词上塌缩为 20/20 → happiness；正文以 GPT 为主、Qwen 作补充（延迟仍可用）。
4. ✅ `weights_from_label.py` 已导出两套 7 维权重。

---

## 第三步：实现 Xpress 对照（克隆官方仓库，只借「选情绪」）✅

1. ✅ 已拉取官方提示相关文件到 `C1/code/baselines/third_party/Xpress/`（稀疏：`Conversational.js` / `realtime_server.js` / `base_emotion_bank.json`；完整 clone 太慢）。  
2. ✅ 提示词抄到 `C1/code/baselines/xpress_conv_prompt.txt`（来自 `realtime_server.js` chunk 反应 system prompt）。  
3. ✅ 不要网页脸；只要标签。  
4. ✅ `run_xpress_conv.py`：转折句 013/014 拆 2 段；输出 emotion + low/high。  
5. ✅ 已跑并映射权重：
   - ✅ GPT → `out/xpress_gpt.csv` + `xpress_gpt_weights/`（22 段 / 20 句，0 error；正文主用）
   - ✅ 本地 Qwen → `out/xpress_qwen.csv` + `xpress_qwen_weights/`（可解析但质量弱，作补充）
   - `xpress_label_map.json`：interest/calm/confusion 等 → neutral；stress/concern → fear 等
   - high→1.0，low→0.5；多段句取**最后成功段**写权重

---

## 第四步：导出本文方法的权重和耗时 ✅

1. ✅ `D/code/export_ours_weights.py`：Qwen CausalLM + E1 en/pole_50 轴/scaler + FKCN + EMA(0.6)。  
2. ✅ 同一 20 句已导出：
   - 逐 token 7 维 → `D/out/precompute/ours_weights/{id}.csv`（含 V/A/D）
   - 整句耗时 → `D/out/precompute/ours_latency.csv`（供 D 参考；正式表仍走第五步 ≥3 次）  
3. ✅ 模型：本地 Qwen2.5-0.5B（conda，CPU）。

---

## 第五步：做延迟表（D，不需要实机）✅

1. ✅ `D/code/run_latency_d.py`：20 句 × 3 次，测「文本就绪 → 表情控制量算完」。  
2. ✅ 五条件：ours / Mishra×GPT / Mishra×Qwen / Xpress×GPT / Xpress×Qwen。  
3. ✅ 汇总：`D/out/latency_table.csv`、`latency_raw.csv`、`machine_env.txt`（当前为开发机预跑；**本地三条件**正式数见第七步工控机补测）。

---

## 第六步：Grimace 迁移示意（C3）✅

**定位：** **不当独立实验小节**；正文叙述中强调「好迁移 / 可部署」，定量仍归 C1。

1. ✅ 目录：`C/C3/` 仅保留正文主图 + `叙述.txt`（撰写要点、图注边界、与 A/B/C1/D 分工）。  
2. ✅ 图：对话关键 token ×（Grimace 上行 / 实机下行）并排，展示同一套 7 维权重换显示端即可。  
3. ✅ 写法：放方法节「与舵机解耦」附近或实机节作示意插图；图注写清上游不变、本图不报统计量。

---

## 第七步 + 第八步：实机 C1/C2 ✅（C1 无锚定全量；C2 人因另卷完成）

**交接文档：** `C/C1/handoff/`；**论文主数据 = 20260731 无锚定场**：
`C/C1/result/multicond_noanchor_20260731/`（80/80 含 Mirror，30fps）。

1. ✅ 无锚定四条件同场录制（ours / mishra_gpt / xpress_gpt / random）+ 同场延迟
   （`D/out_robot_noanchor_20260731/`，Intel N100 CPU）。
   20260730 场因执行端句级原型锚定（β=0.8，displayed = proto + β(raw−mean)）压扁句内效价动态而弃用，
   先试录 5 句验证锚定关闭后续录全量。
2. ✅ 分析：`C1/code/run_c1_multicond_compare.py` → `C1/output/multicond_noanchor_20260731/c1_multicond_summary.md`。
   - **协议**：四条件统一固定滞后 0.5s；方向命中按 trial 聚类 bootstrap（重叠窗不是独立样本）；
     指令裁剪到可渲染 [-1,1]；Mirror 读数经 40 段静止脸独立校准（效价压缩约 5x + 唤醒串扰）。
   - **结论**：ours 效价 87.5%（CI [76.7, 96.0]，p<1e-4）、唤醒 69.6%（p=0.001）、R² 增益 +0.22/+0.19；
     三基线两维度全机会水平、增益 ≈0。**单场即可支撑 C1，不再分旧场/新场两套口径。**
   - 主图：`C1/code/plot_c1_multicond_paper.py` → `C1/output/multicond_noanchor_20260731/paper/`。
3. ✅ C2 人因（重做口径：按目标情绪回复；T2 单选 token/turn/random；n=20）：
   结果 → `C2/output/human_study_20260731/`（T1 85.5% vs 预设 90.5%；T2 ours 83.5% / turn 5% / random 11.5%）；
   已写入正文 `tab:c2_human`（无独立大图）。本机 `session_noanchor` / survey 包未用于终稿问卷，已删。
4. 历史数据（20260717 旧场、20260730 锚定场、试录、旧延迟）→ `_archive_obsolete_20260731/`。

---

## 第九步：写回论文

1. ✅ Related Work 已点名 Mishra / Xpress。
2. ✅ 实机章节（IV-C）：C1 四条件主图 + 延迟三线表 + C2 人因小表（叙述嵌入，无独立大图）。
3. ✅ 贡献已含 0.7ms 延迟与人因一句。
4. ⏳ C3 迁移示意图插入方法/实机叙述（若尚未进 tex）。
5. ⏳ Limitation / 开源声明收尾；再核对每条贡献是否都有对应图表。

---

## 类别对齐（写脚本时用）

我们的 7 类顺序：`neutral, happiness, surprise, fear, sadness, disgust, anger`（见 `src/grimace_utils.py`）。

| 对照标签 | 映射到 |
|----------|--------|
| Happy / joy | happiness |
| Sad | sadness |
| Fear | fear |
| Anger / angry | anger |
| Surprise | surprise |
| Disgust | disgust |
| Neutral | neutral |
| Xpress 其它（interest 等） | 最接近类或 neutral |

---

## 实验做完后备忘（写正文 / 投稿前核对）

主 claim 在 A/B + C1 对照 + C3 + D 补齐后应能站住。还要注意：

1. **写进 tex**：C 对照、C3 图、D 表；贡献补延迟数字；摘要「几组实验」与终稿一致。  
2. **页数**：RAL 正文最多 8 页（含图、表、参考文献；附录也算页）。视频只能进 Multimedia，不能当加页附录。  
3. **对照怎么写**：比的是接到我们 7 类预置后的控制粒度与延迟，不是复现原 Furhat/网页脸；Qwen 跑提示词若差，以 GPT 为主、Qwen 作补充。  
4. **C2**：问卷若已发出，人因对 SOTA 偏弱，客观结论靠 C1+D，文中说明 turn 仅为近似。  
5. **C3 边界**：只证明能换到 Grimace，不写成「任意实体机器人都已验证」。  
6. **措辞**：有了 D 就删掉「附加延迟可忽略」，改成具体毫秒；神经科学/修辞别写成已证明。  
7. **结果风险**：对照要拉开差距，刺激句保住转折/讽刺子集。  
8. **收尾**：Limitation、引用格式、代码/数据是否开源，文末一句说清。
