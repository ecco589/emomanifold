# FKCN Visualizer (M3test)

交互式展示 **(V, A) → FKCN 7 维权重 → Grimace 表情**。

## 启动

```bash
python M3test/FKCN_visualize/server.py
```

浏览器打开 http://127.0.0.1:8765/

## 操作

- 左侧：V–A 平面权重场（红色透明度 = 情绪激活强度），点击选点
- 右上：输入 Valence / Arousal（−1 … 1）
- 右侧：显示开关 + 7 维权重条
- 右下：Grimace 实时渲染

逻辑与 `src/fkcn.py`、`src/grimace_utils.py` 一致。
