# 视频 VA 值导出系统

这是一套本地离线流程：你输入一个视频，系统用 Py-Feat `Detectorv2` 逐帧识别人脸表情，并导出 Valence / Arousal（VA）表格、曲线图和汇总报告。

## 1. 第一次使用

双击：

```text
setup_env.bat
```

它会创建 `.venv` 虚拟环境并安装依赖。第一次真正分析视频时，Py-Feat 还会联网下载模型权重，所以第一条视频会慢一些。

安装后可以双击：

```text
diagnose.bat
```

它会检查 Py-Feat、Torch、OpenCV、FFmpeg 等依赖是否能正常加载。

## 2. 导出视频 VA

最简单的方式：

1. 把视频拖到 `run.bat` 上。
2. 如果没有“原系统 VA CSV”，第二个输入直接按 Enter。
3. 结果在 `results/视频名_时间/` 下面。

也可以双击 `run.bat`，手动粘贴视频路径。

## 3. 输出文件

每次运行会生成一个独立结果文件夹，里面主要有：

```text
video_va.csv              每一帧 VA，推荐首先看这个
video_va_full.csv         Py-Feat 完整输出，含 AU、表情类别、姿态等
video_va_curve.png        VA 曲线图
video_va_report.xlsx      Excel 报告
summary.json              运行配置和有效识别比例
```

`video_va.csv` 的核心列：

```text
frame                     视频帧号，从 0 开始
timestamp_s               秒级时间戳
face_score                人脸检测置信度
face_valid                这一帧是否达到有效阈值
valence_raw               模型原始 Valence
arousal_raw               模型原始 Arousal
valence                   通过 face_score 阈值后的 Valence
arousal                   通过 face_score 阈值后的 Arousal
valence_filled            对短暂漏检做线性补洞后的 Valence
arousal_filled            对短暂漏检做线性补洞后的 Arousal
valence_smooth            平滑后的 Valence，建议用于对比和画曲线
arousal_smooth            平滑后的 Arousal，建议用于对比和画曲线
```

## 4. 和原系统 VA 对比

如果你有原系统导出的 VA CSV，可以在 `run.bat` 第二个输入里填这个 CSV 路径。系统会额外生成：

```text
compare_aligned.csv       按时间戳对齐后的逐点对比
compare_metrics.csv       MAE、RMSE、相关系数、CCC 等指标
compare_curve.png         原系统 VA 与视频识别 VA 的曲线对比
```

原系统 CSV 最好包含这些列名之一：

```text
timestamp_s / time_s / seconds / time / t / timestamp_ms / frame
valence / v / original_valence / target_valence
arousal / a / original_arousal / target_arousal
```

如果原系统 CSV 用 `frame` 作为时间列，需要命令行传入 `--source-fps`。

## 5. 命令行高级用法

```powershell
.venv\Scripts\python.exe video_va_pipeline.py `
  --video "D:\videos\robot_test.mp4" `
  --out-dir results `
  --device auto `
  --batch-size 1 `
  --frame-step 1 `
  --face-threshold 0.70 `
  --smooth-ms 200
```

带原系统 VA 对比：

```powershell
.venv\Scripts\python.exe video_va_pipeline.py `
  --video "D:\videos\robot_test.mp4" `
  --source-va "D:\videos\source_va.csv" `
  --time-offset-ms 0
```

常用参数：

```text
--device auto/cpu/cuda/mps     自动或指定 CPU/GPU
--batch-size 1                 CPU 建议 1；显卡可试 4 或 8
--frame-step 1                 每 1 帧处理一次；调大可加速但不是逐帧
--face-threshold 0.70          有效人脸置信度阈值
--smooth-ms 200                VA 曲线平滑窗口
--interpolate-limit-ms 250     短暂漏检补洞上限
--no-excel                     不生成 xlsx
```

## 6. 视频录制建议

第一版先用稳定、普通的视频格式：

```text
1080p
30 fps
MP4 / H.264
关闭 HDR、美颜、人像模式、电影模式、滤镜、自动增强
横屏固定机位
脸部占画面宽度约 30% 到 60%
画面里尽量只有一个脸
```

如果手机录的是 HEVC/H.265 或 HDR，OpenCV/Py-Feat 可能打不开，建议先转成普通 MP4/H.264。

## 7. 重要限制

Py-Feat `Detectorv2` 是面向人脸训练的。仿真度高的机器人脸可能可用，但机械脸、屏幕脸、卡通脸可能检测不到，或者 VA 数值没有稳定语义。正式实验前，先录 5 到 10 秒短视频测试 `summary.json` 里的 `valid_ratio`。

另外，Py-Feat 工具包本身和内部模型权重许可证不完全相同。官方文档说明 `Detectorv2` 的 `face_multitask_v2` 权重为非商业研究用途；如果以后要做商业部署，需要重新确认模型授权或替换模型。

## 8. 常见问题：结果文件夹是空的

如果运行时看到类似：

```text
Could not load libtorchcodec
FFmpeg is not properly installed
On Windows, ensure you've installed the "full-shared" version
```

这不是视频识别结果为空，而是 Py-Feat 还没启动成功。原因通常是 Windows 没有安装 TorchCodec 需要的 shared FFmpeg DLL。

可以双击：

```text
install_ffmpeg_hint.bat
```

或者手动执行：

```powershell
winget install --accept-source-agreements --accept-package-agreements Gyan.FFmpeg.Shared
```

安装后关闭当前窗口，重新打开 `run.bat`。如果仍然失败，先运行 `diagnose.bat` 看具体依赖状态。
