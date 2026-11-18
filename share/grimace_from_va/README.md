# Grimace from VA

把 valence / arousal 画成 Grimace 漫画脸。映射和渲染与仓库里的实验脚本一致：

`RAL/experiment/C/C1/code/render_grimace_from_source_va.py`

本目录可以单独拷走，不依赖仓库其余代码，也不需要 PyTorch。

## 管线

```
VA（裁剪到 [-1, 1]）
  → FKCN，13 个原型，7 维权重
  → EMA，α = 0.6
  → 10 fps，150 ms smoothstep，单帧每维最多变化 0.05
  → 去掉 neutral，得到 6 维
  → Grimace 画到白底，ffmpeg 合成 MP4
```

6 维顺序固定为：

| 下标 | 名字 |
| --- | --- |
| 0 | joy |
| 1 | surprise |
| 2 | fear |
| 3 | sadness |
| 4 | disgust |
| 5 | anger |

7 维 FKCN 顺序是 `neutral, happiness, surprise, fear, sadness, disgust, anger`。Grimace 的 joy 对应 happiness，neutral 不传给脸。

默认渲染会把当前帧的最大权重拉到接近 1，放大倍数不超过 2。实验视频用的就是这个。要按 FKCN 原始混合比例画，加 `--no-amplify`。

单张静帧走 `va_to_weights`：不做 EMA，也不从全零淡入。视频走上面的整条 display path，第一帧从全零权重缓入。所以同一点的静帧，和视频里该点的第一帧，不会相同。

## 安装

需要 Python 3.9+、ffmpeg。

```bash
cd grimace_from_va
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
python -m grimace_from_va --self-test
```

`--self-test` 只检查 VA 映射，不开浏览器。

## 命令行

一个点，打印 6 维权重：

```bash
python -m grimace_from_va --valence 0.81 --arousal 0.66
```

写成 PNG：

```bash
python -m grimace_from_va --valence 0.81 --arousal 0.66 -o happy.png
```

一条轨迹写成 MP4。示例文件是 `example/sample_va.csv`。

```bash
python -m grimace_from_va --csv example/sample_va.csv -o out.mp4
```

只导出每帧权重，不开浏览器：

```bash
python -m grimace_from_va --csv example/sample_va.csv --weights-out frames.csv
```

每个输入行经过 EMA 之后的权重（没有插值）：

```bash
python -m grimace_from_va --csv example/sample_va.csv --rows-out rows.csv
```

和 C1 实验视频对齐时，首尾各停 1 秒：

```bash
python -m grimace_from_va --csv va.csv --lead 1 --trail 1 -o match_c1.mp4
```

常用参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--alpha` | 0.6 | EMA 系数 |
| `--lead` / `--trail` | 0 | 首尾停留秒数 |
| `--width` `--height` | 640 | 画面像素 |
| `--no-amplify` | 关 | 不拉高最强权重 |
| `--no-prefer-raw` | 关 | 即使有 `raw_valence` 也读 `valence` |

## CSV

表头至少要有 `valence` 和 `arousal`。

如果同时有 `raw_valence`、`raw_arousal`，默认读 raw。实验导出的 `source_va.csv` 就是这种：`raw_*` 是投影坐标，`valence` / `arousal` 有时已经平滑过。

可选列：

- `timestamp_s`（也认 `time_s`、`time`）。没有时间列时，相邻点按 150 ms 间隔补上。
- `token`（也认 `text`）。只写进 `--rows-out`，不参与画脸。

时间必须非递减。数值超出 [-1, 1] 会被裁剪。

## Python 接口

在本目录下运行，或把本目录加进 `PYTHONPATH`。

```python
from grimace_from_va import (
    va_to_weights,
    map_rows,
    display_trajectory,
    GrimaceRenderer,
    render_png,
    render_video,
)

# 单点。返回 [joy, surprise, fear, sadness, disgust, anger]
weights = va_to_weights(0.81, 0.66)

# 静帧
render_png(0.81, 0.66, "happy.png")

# 一行一个脸：FKCN + EMA，不做帧间插值
mapped = map_rows(valence, arousal, timestamps, alpha=0.6)
mapped.weights          # (N, 6)
mapped.fkcn             # (N, 7)，含 neutral
mapped.save_csv("rows.csv")

# 实验用的 10 fps 轨迹
traj = display_trajectory(
    valence,
    arousal,
    timestamps,
    alpha=0.6,
    lead_s=0.0,
    trail_s=0.0,
)
traj.weights            # (帧数, 6)
traj.times              # 从 0 开始的秒
traj.fps                # 10
traj.rate_limit_fraction
traj.save_csv("frames.csv")

# 自己画。GrimaceRenderer 要在打开它的那个线程里调用。
with GrimaceRenderer(width=640, height=640, amplify=True) as renderer:
    png = renderer.render(weights)          # bytes
    renderer.save(weights, "face.png")

# CSV → MP4。lead_s=1, trail_s=1 对齐实验视频。
render_video("va.csv", "out.mp4", weights_out="frames.csv")
```

`display_trajectory` 还有这些参数，默认值和实验脚本相同：

| 参数 | 默认 |
| --- | --- |
| `fps` | 10 |
| `transition_ms` | 150 |
| `max_delta` | 0.05 |
| `settle_frames` | 10 |

最后一个点最多再走 10 帧把插值收完。前面的点如果下一个点来得早，过渡会被截断。

## 目录

```
grimace_from_va/
  README.md
  requirements.txt
  example/sample_va.csv
  grimace_from_va/
    __init__.py          对外接口
    __main__.py          命令行
    fkcn.py              VA → 7 维
    trajectory.py        EMA 和 10 fps 轨迹
    renderer.py          Playwright 出图、ffmpeg 合成
    vendor/grimace.umd.js
    vendor/LICENSE.grimace
```

Grimace 本体是 MIT 许可，见 `vendor/LICENSE.grimace`。上游：<https://github.com/grimace-project/web>
