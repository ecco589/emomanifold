# NRC VAD Lexicon（不随仓库分发）

重建词库和部分论文实验会读取 NRC Valence, Arousal, and Dominance Lexicon。该词表由加拿大国家研究委员会 Saif M. Mohammad 发布，条款允许非商业研究与教学使用，并明确禁止再分发数据。因此本目录不包含词表文件。

请从词表主页自行下载，并按官方包的目录结构放在这里：

http://saifmohammad.com/WebPages/nrc-vad.html

本仓库脚本使用的路径：

- `NRC-VAD-Lexicon.txt`
- `OneFilePerLanguage/Chinese-Simplified-NRC-VAD-Lexicon.txt`
- `BipolarScale/NRC-VAD-Lexicon.txt`

下载后不要把这些文件提交回仓库。`.gitignore` 已忽略本目录中除本说明以外的文件。

运行时不需要整份词表。流式投影读取的是 `data/vad_lexicon/` 里已经抽好的极词 JSON。这些 JSON 里的分数来自 NRC VAD，使用时请引用：

Saif M. Mohammad. Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. ACL 2018.
