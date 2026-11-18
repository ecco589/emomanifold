# 改稿计划

骨架：Emo-H2A（范式）→ EmoManifold（实现）→ 主实验（跟踪、延迟）→ 机制验证 → 人因放最后。
三条原则是设计要求，不是三节的标题。DexViTac 也如此：原则写在 Hardware Design 开头，节名仍是具体做法。原则按「表情有多少、投影有多细、路径有多短」；三节按流水线：建轴、按 token 投影、映射到脸上。training-free 只留在相关工作里（不用语料前向或再训练），不单列成第四条原则，也不写进引言第三段的三个缺口。
全文只用 **EmoManifold**。缩写用 **H2A**，全称 hidden-to-actuation 只在引言第四段写一次。第一段可以文学，其余按 RA-L：对象、做法、结果。

结构照 DexViTac 的写法，不借用它的内容：引言用「缺口 → To address these challenges, we propose → 结果 → 开源句 → In summary, our main contributions are」；贡献是「加粗名: 一句话」。方法开头先给三条原则（短标签 + 一句），再用 first / next / finally 点出三节。实验开头是三问，按「效果 → 时机 → 信号」分层。

## 1 题目

`Emo-H2A: Driving Robot Faces Directly from LLM Hidden States at Token Rate`

原引语 “Written All Over the Face” 移到引言第一段。H2A 的全称 hidden-to-actuation 在第四段首次提出范式时写出。

## 2 引言

四段：自然性 → 核心瓶颈 → 三个具体瓶颈 → 范式与结果。

1. 神经科学（现 L52）压成开头 1–2 句，收到交互自然性。
   > “It's written all over your face.” As we speak, emotion surfaces on the face continuously, word by word, long before we name it. A machine that converses with us as naturally as another person—one whose face moves with what it is saying—has long been one of robotics' most human aspirations. Large language models and high-DoF robotic faces now give us a chance to take a real step toward it.

2. 第一句点出瓶颈，不写延迟和连续表情。然后两类工作各一句：LLM 表情系统把表情当成事后标签；隐空间工作证明情感线性可读，但没有接到脸上。收束：都没有把内部情感直接接到脸上。
   > Current robot faces are driven by a discrete emotion label, not by the LLM's internal affective state during generation.

3. 承接第二段，从「内部情感没有接到脸上」收紧到三个具体缺口，逐级深入，并进入后文的三条贡献：表情连续、token 级响应、毫秒级延迟。每条写成「上一类工作解决了上一层，但没有解决下一层」。对照原文，不硬塞。training-free 不放进这三个缺口。
   - 连续情感空间已经有人做过（如 Kismet），但没有接到语言模型的生成过程。
   - LLM 驱动的脸（Mishra、Xpress）接到了生成结果，但仍是整句一个离散标签，没有句内连续变化。
   - 回合级方法即便把标签做得更细，也要等整句结束才更新，做不到生成过程中的低延迟响应。

   | 维度 | 现状 | 本文 |
   |---|---|---|
   | 表情表示 | 离散标签（Mishra 6 类；Xpress 类别+强度） | 连续 VAD |
   | 响应粒度 | 话轮级 | token 级 |
   | 响应延迟 | 等整句结束才更新 | 生成过程中即可更新；0.7 ms 不写进这一格 |

4. 提出范式、实例和结果。没有可开源的硬件设计或数据集。贡献前只写代码：*To foster further research, we will open-source the code. In summary, our main contributions are:*
   > We propose **Emo-H2A** (hidden-to-actuation), which maps LLM hidden states to facial actuator commands during decoding and removes the intermediate emotion label. To our knowledge, it is the first paradigm to close this loop on a physical face. We instantiate it as **EmoManifold**, a training-free projection and mapping pipeline. On a 29-DoF robot face, token-level actuation follows within-utterance affect relative to turn-level discrete labels, and the face updates during generation rather than only after the reply is complete.

5. 贡献三条，关键词加粗：
   1. **Emo-H2A**: a paradigm that takes the hidden-state feature at each decoding step directly to facial motor commands.
   2. **EmoManifold**: a method that constructs a valence–arousal–dominance subspace and projects each token's hidden state onto it.
   3. **Physical evaluation**: on a 29-DoF face, token-level actuation follows within-utterance affect relative to turn-level discrete labels, and the face updates during generation rather than only after the reply is complete. 人因偏好不写进这句。

6. 反讽比例、关键词命中率、Spearman、R² 只出现在实验后半的机制验证里，不进摘要、第四段和贡献。0.7 ms 是每 token 的计算开销；首表情时间才是延迟贡献。两者不要写成同一个数。

## 3 相关工作

按系统分类，不按实验完成顺序。每节末一句点出与本文的差别，不把引言的论证再讲一遍。

1. **Robotic Facial Expression Generation**：FACS、Kismet、Emo、X2C/Ameca。
2. **LLM-Driven Expressive Systems**：FurChat、SAFE、Nadine、Expressive Furhat；基线 Mishra、Xpress。
3. **Linear Affective Representations in Latent Space**：Russell、NRC-VAD、Tigges、Reichman。差别：他们要语料前向或训练模块，本文只用静态 unembedding 矩阵。

## 4 方法

方法第一句点名一次：*This section specifies EmoManifold, our instantiation of Emo-H2A.* 全称不重复，引言第四段已经写过。三节正文不再重复这两个名字。首页通栏是效果图 Fig. 1（原 Fig. 7），正文在提出结果处指向它。系统总览是 Fig. 2，放在方法开头，正文只留 *Fig. 2 overviews the system.* 图注用一句 Overview，不逐个复述模块。不新起组件名。

三条原则放在路线图之前。句式照 DexViTac：标签点出要求，冒号后用祈使句。原则名不复用为节标题。

- **Expression-Rich**: Use a continuous space of expressions rather than a single label, so slight changes in affect remain representable.
- **Token-Level**: Update the face at each token by projecting the corresponding hidden state at that decoding step.
- **Direct-Readout**: Derive the facial command from one projection, and introduce no reasoning step that would add latency.

路线图按流水线写，不按原则的顺序重排：first 建轴（III-A），next 按 token 投影（III-B），finally 映射到脸上（III-C）。三节合在一起满足上面三条。

三节标题用做法，不写成 From X to Y：

- **III-A Training-Free Affective Axes**：离线建一次轴，在线只在静态 unembedding 上做投影。不让模型先说出情绪，也不另跑分类器。Pole-word clusters / Axis construction / AdaptiveScaler。
- **III-B Per-Token VAD Projection**：每个解码步，包括第一个 token，都做这次投影。Hidden-state projection / Runtime cost。0.7 ms 是这一步的计算开销，写在这里，不写进原则句。
- **III-C Continuous Facial Mapping**：效价和唤醒度合成执行器权重。预设是基底，表情不收成一张类别表。VA-to-expression mapping / Preset fusion / Smoothing and rate limiting。

## 5 实验

不重做实验，也不按方法的工程顺序（先轴、再投影、再上脸）写。开头三问分层，写的是要验证的问题，不预写结论。人因不单列成一问。连续表情和 token 级更新仍是同一次实机对比里的两个方面，不拆开。

- **Q1. Within-Utterance Tracking:** Does token-level actuation follow affect within a sentence, relative to turn-level discrete labels?
- **Q2. Response Timing:** Does the face update during generation, rather than only after the reply is complete?
- **Q3. Affective Readout:** When the surface words are held fixed, does the readout change with the context?

Q1 是主效果，Q2 是这次更新发生的时机，Q3 问驱动脸的信号是不是语境中的情感，而不是词面。Q3 用已有的子空间和反讽实验回答，不新做实验。

- **IV-A Setup**：平台、条件、指标、下表。
- **IV-B Main Comparison**（Q1–Q2）：与回合级方法的实机对比。Actuation fidelity（Fig. 6）回答 Q1；On-device latency 回答 Q2。
- **IV-C Mechanism**（Q3）：先看情感轴是否稳定（原 A），再看同一批词在不同语境下读出是否改变（原 B）。
- **IV-D Robustness**：Hyperparameter robustness / Cross-carrier transfer（指回首页 Fig. 1）。
- **IV-E Human Study**：识别与偏好，放在主对比之后，不作开头的问题。

| | EmoManifold | Mishra et al. | Xpress |
|---|---|---|---|
| 响应粒度 | token | 话轮 | 话轮 |
| 连续表情空间 | ✓ | ✗（6 类） | △（类别+强度） |
| 不输出情感标签 | ✓ | ✗ | ✗ |
| 无外部 API | ✓ | ✗ | ✗ |
| 首 token 即更新 | ✓ | ✗ | ✗ |
| 句内转折 | ✓ | ✗ | ✗ |
| 实机验证 | ✓ | 核对原文后填 | 核对原文后填 |

不单列“免训练”：两个基线也是提示工程，不训练。

## 6 结论

本轮不改 Discussion 与 Conclusion。

首页排版照 SensorPerch：Fig. 1 用 `\IEEEaftertitletext` 放在标题和摘要之间，通栏，内容是当前 `fig7.png`（一句回复里四个划线 token 上的仿真脸和实机脸）。效价–唤醒度曲线还没画进这张图，图注先不写曲线。Fig. 2 是系统总览，`figure*` 放在方法节开头，不放首页。稳健性一节指回 Fig. 1，不再单放一张双载体图。参考文献按 `suggest_detail.md` 第 12 条改。投稿前检索 “H2A” 和 “first hidden-state-to-face” 是否已被占用，有冲突就在首次出现写全称并收窄 “first” 的范围。
