# RAL 实验：对照与补充（按顺序做）

目录地图见 [`STRUCTURE.md`](STRUCTURE.md)。

---

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

## 第七步 + 第八步：实机 C1/C2 同场重跑 ✅（无锚定重录完成；C2 问卷待裁切发放）

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
3. ⏳ C2 问卷：从 `C2/result/session_noanchor_20260731/` 裁切打包。
4. 历史数据（20260717 旧场、20260730 锚定场、试录、旧延迟）→ `_archive_obsolete_20260731/`。

---

## 第九步：写回论文

1. ✅ Related Work 已点名 Mishra / Xpress。  
2. ✅ 实机小节（IV-C）已写入：镜像测试主表（旧场，trial 聚类）、四条件对照表（新场修正协议）、
   实机延迟段（per-token 0.7ms / TTFF 1.1s vs 基线 13–30s）。  
3. ⏳ 贡献里补上延迟数字；C3 迁移示意图插入方法/实机叙述。  
4. ⏳ 检查每条贡献都有对应表或图。

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
