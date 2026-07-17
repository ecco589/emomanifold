# 原系统 VA CSV 示例

`source_va_example.csv` 是可选对比文件的最小格式。

推荐列名：

```text
timestamp_s,valence,arousal
```

如果你的原系统是按 token 输出 VA，也需要给每个 token 一个时间戳；否则视频识别出来的逐帧 VA 没法和原系统 VA 做可靠时间对齐。
