# Fig. 3 作图交接

论文：EmoManifold（RAL 英文稿）。图号 Fig. 3，跨语言情感几何。

现图见 `current_fig3.png`。请按下面的数据重绘，不要沿用现图的配色和排版。

## 图里有什么

两栏。

**(a) 左：示意图，不是实测。** 三条情感轴（Valence / Arousal / Dominance）和一组理想正交基。五个情绪词簇在不同语言里落在一起。仓库里没有保存每个词的三维坐标，不要把左图画成散点实测图，也不要给词标注数值。

现图左栏的标签（Joy、Fear、Sadness、Anger，以及放大镜里的 “EN: Joy / CN: 快”）和实验不一致。实验的五簇是：恐惧、舒适、支配、力量、平静。见 `emotion_clusters.csv`。

**(b) 右：唯一需要按数作图的部分。** 现图是一张表，只含英文极词构轴（N = 50）的结果。数字用三位小数，来源见 `panel_b_as_plotted.csv`。

| 列 | 含义 |
|---|---|
| \|cos VA\|、\|cos AD\|、\|cos VD\| | 两条轴的绝对余弦。越小越接近正交。箭头向下。 |
| Cohesion | 五个情绪簇的簇内平均两两余弦，再对五簇取平均。越大表示跨语言同义词越聚在一起。箭头向上。 |
| (random) | 同一模型上、随机三维正交基的对照，括号内。100 次试验的簇内余弦再取平均。 |

模型列：Qwen2.5-0.5B，Llama-3.2-3B。

## 正文里有、现图里没有

中文极词构轴的同样四行指标在 `metrics_full.csv`，`in_current_figure = no`。除非另行要求，不要加进这张图。

## 文件

| 文件 | 内容 |
|---|---|
| `panel_b_as_plotted.csv` | 现图右表上的数字（三位小数） |
| `metrics_full.csv` | 未四舍五入的轴间余弦与五簇平均凝聚度，含中文极词 |
| `cluster_cohesion.csv` | 五簇各自的凝聚度和该簇的随机基线 |
| `emotion_clusters.csv` | 五簇、语言、词 |
| `llama_skipped_words.csv` | Llama 分词器切不开、未进入凝聚度的词 |
| `current_fig3.png` | 现在的图，只作对照 |

Qwen 的五簇词全部进入计算，没有跳过的词。

## 指标怎么来的

每个模型用自己的解嵌入矩阵、各自用 50 个极词构三条轴。轴间 \|cos\| 是轴向量两两绝对余弦。

凝聚度：把簇内每个词（必须是该模型的一个 token）投到这三条轴上，得到三维向量，再算簇内所有词对的余弦平均。图上的 Cohesion 是五个簇的平均。随机基线用随机正交基，不做标定缩放，每簇 100 次，再对五簇平均。Qwen 的随机种子是 42。

五簇内部编号与论文用词：terror = fear，comfort = comfort，boss = dominance，strength = strength，zen = calm。
