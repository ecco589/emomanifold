# EmoManifold

EmoManifold（文稿中的系统名是 Emo-H2A）从大语言模型的隐状态读取逐 token 的连续情感，并驱动一张脸。它不要求模型说出情绪标签。流式解码时取出末层隐状态，投影到预先从极词和解嵌入矩阵构造的效价、唤醒度、支配度（VAD）轴上，再用 FKCN 把效价–唤醒度映射成七种基本表情的混合权重，交给 Grimace 画成漫画脸。若本机另有 29 自由度面部驱动，同一组权重也可以下发到实机。

项目页：<https://ecco589.github.io/emomanifold/>

仓库根目录的代码以 MIT 许可证发布，版权人 Yige Liu，2026。见 [LICENSE](LICENSE)。第三方组件见 [NOTICE](NOTICE)。

## 这条管线做什么

1. 用情感极词在模型的解嵌入矩阵上定出 VAD 轴（`src/dimension_extractor.py`）。
2. 每个新 token 的隐状态投影到这些轴上（`src/emotion_projector.py`）。
3. `AdaptiveScaler` 用 `data/training_corpus.txt` 拟合缩放，把投影值收到可用范围。
4. FKCN 把 VAD 收成七维权重：neutral、happiness、surprise、fear、sadness、disgust、anger（`src/fkcn.py`）。
5. Grimace 按权重画脸。可选的 `RobotFaceSink` 在导入到本地 `face` 包时，把同一组权重融合成 29 个自由度的指令。

`config.py` 里登记了四个模型名：`qwen`（流式默认，Qwen2.5-0.5B-Instruct）、`ernie`（ERNIE-4.5-0.3B-PT）、`deepseek`、`glm`。用环境变量 `EMOTION_MODEL` 切换。权重文件不在仓库里，需要自行下载到 `models/`，或让加载逻辑使用本机已有缓存。

## 仓库布局

```
├── src/                  投影、FKCN、流式生成、Grimace 与实机接口
├── scripts/              下载模型、训练 scaler、测试、渲染、词库、实验机部署
├── examples/             Gradio 流式界面
├── web/                  网页演示（GitHub Pages）
├── data/                 运行用的极词 JSON、训练语料；NRC 词表需自行下载
├── share/grimace_from_va/  只吃 VA 坐标的独立小工具，不需要 PyTorch
├── grimace-main/         上游 Grimace 引擎，保留其 MIT 许可证
├── docs/ROADMAP.md       里程碑备忘
└── RAL/                  文稿与实验记录，运行软件不必进入
```

`RAL/versions/` 是更早的文稿快照。`RAL/notes/` 是起草备忘。两者都不是运行入口。

## 环境

- Python 3.9 及以上。完整管线需要 PyTorch，能用 CPU；苹果 GPU 会走 MPS，有 NVIDIA GPU 时走 CUDA。
- 网页演示需要 Node.js 20。
- 把 VA 画成 PNG 或 MP4 时需要 Playwright 的 Chromium，视频还需要 ffmpeg。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bash scripts/install_dependencies.sh` 做的是同一件事。表情渲染再执行一次 `bash scripts/install_face.sh`。

## 运行

### 不下载语言模型：由 VA 画脸

`share/grimace_from_va` 可以单独使用。`--self-test` 只检查映射，不打开浏览器。

```bash
cd share/grimace_from_va
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m grimace_from_va --self-test
python -m grimace_from_va --valence 0.81 --arousal 0.66
```

写出 PNG 前需要 `python -m playwright install chromium`。从 CSV 合成 MP4 还需要本机有 ffmpeg。用法见该目录的 README。

### 网页演示

演示页把对话发给推理 API，自己只负责把返回的 VAD 画成 Grimace 脸。

```bash
cd web
npm ci
# 默认请求本机 8788。改地址时设置 VITE_API_URL。
npm run dev
```

已部署的页面在 <https://ecco589.github.io/emomanifold/>。构建时的 API 地址来自环境变量 `VITE_API_URL`（见 `.github/workflows/deploy-pages.yml`）。`web/va-editor.html` 是实验用的 VA 轨迹编辑器，开发服务器会读 `RAL/newmainexp/` 里的试次 CSV。

### 本机推理 API

API 按 `config.py` 加载模型，并在 `POST /v1/chat/stream` 上流式返回 token 和 VAD。默认端口 8788。

```bash
pip install -r requirements-api.txt
# 按 config.py 把模型放到 models/，或事先放进本机缓存
EMO_DEVICE=cpu python scripts/lab/api_server.py
```

另开一个终端，在 `web/` 里执行 `npm run dev`，浏览器打开 Vite 打印的地址。

`scripts/lab/` 里其余脚本通过 SSH 把 API 或模型文件同步到另一台机器。SSH 主机名用环境变量 `EMO_LAB_SSH` 指定，默认只是本机 SSH 配置里的一个别名。这些脚本不是本地试用所必需的。

### Gradio 流式界面

这个界面在进程内加载模型，逐 token 显示 VAD，并用 Grimace 画脸。勾选「实机」时才会尝试 29 自由度驱动。

```bash
source .venv/bin/activate
python examples/stream_ernie_vad_ui.py
```

macOS 也可以双击 `examples/launch_ernie_vad_ui.command`。它使用当前虚拟环境里的 Python，不再写死某台电脑上的解释器路径。启动后打开 <http://127.0.0.1:7860>。

### 投影测试与 scaler

`scripts/test_with_scaler.py` 用 ERNIE 对四句中文做投影，并把图写到 `outputs/ernie/`。需要本机已经有对应模型和 `outputs/scalers/ernie_scaler.json`。还没有 scaler 时，先准备好 `data/training_corpus.txt`，再运行 `python scripts/train_scaler.py`。

`python scripts/download_model.py` 会按 `config.py` 里的当前模型名，用 ModelScope 下载权重。

## 实机 29 自由度

`src/robot_face_sink.py` 会从仓库根目录下的 `face/` 导入 `face_driver`。这个硬件驱动不在本仓库中。没有该目录时，Gradio 和流式管线仍然可以做投影和 Grimace 渲染；只有打开实机开关才会报驱动缺失。

## 词库

`data/vad_lexicon/` 里的三个 JSON 是运行时用的中文极词表，分数来自 NRC VAD Lexicon。NRC 条款禁止再分发整份词表，所以 `data/NRC-VAD-Lexicon/` 只保留了获取说明。重建词库时，自行下载后运行 `python scripts/update_lexicon_from_nrc.py`。使用这些分数请引用 Mohammad, ACL 2018。

## 论文与实验

文稿、图和实验记录在 `RAL/`，索引见 [RAL/README.md](RAL/README.md)。这里的数字和结论以文稿为准，本 README 不转述实验结果。

`docs/ROADMAP.md` 是开发备忘。其中提到的里程碑标签没有发布在这个远程仓库上，不能用 `git checkout` 某个标签回到旧快照。

## 许可证与第三方

- 本仓库原创代码：MIT，Copyright (c) 2026 Yige Liu。
- Grimace：MIT，许可证在 `grimace-main/LICENSE`，上游 <https://github.com/grimace-project/web>。
- NRC VAD Lexicon：研究使用、禁止再分发。未随此树提供。
- `RAL/ieeeconf/` 与 `RAL/IEEEtran.bst`：IEEE / Michael Shell 的 LaTeX 模板，见文件内版权声明。
