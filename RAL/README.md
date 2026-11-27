# 论文与实验

这里是 Emo-H2A / EmoManifold 的文稿和实验记录。运行软件不需要进入这个目录。软件入口见仓库根目录的 README。

当前英文文稿标题是 *Emo-H2A: Driving Robot Faces Directly from LLM Hidden States at Token Rate*。中文稿是 `emotion_manifold.tex`，英文稿是 `emotion_manifold_en.tex`。对应 PDF 放在同目录。作者栏在这两份当前稿里仍是占位符。

| 路径 | 内容 |
| --- | --- |
| `emotion_manifold.tex`、`emotion_manifold_en.tex` | 当前文稿 |
| `fig/` | 文稿插图 |
| `references.bib`、`references_en.bib` | 参考文献 |
| `ieeeconf/`、`IEEEtran.bst` | IEEE 模板，许可证见文件头，不属于仓库根目录的 MIT |
| `experiment/` | 实验脚本、结果和各子目录说明 |
| `material/` | 实验材料（例如讽刺三元组） |
| `newmainexp/` | 主实验试次与 VA 轨迹；网页里的 VA 编辑器会读这里 |
| `video/` | 文稿视频素材 |
| `outline.md` | 投稿大纲 |
| `versions/` | 更早的文稿、arXiv 源包和 PDF 快照，不是运行软件所需 |
| `notes/` | 起草备忘，不是运行软件所需 |
| `handoff/` | 作图交接文件 |

在 `RAL/` 下编译英文稿：

```bash
xelatex emotion_manifold_en.tex
```

中文稿需要可用的 CJK 环境。实验怎么跑，以 `experiment/` 各子目录的 README 为准。部分实验脚本会读取 `data/NRC-VAD-Lexicon/`，该词表需自行下载，见该目录中的说明。
