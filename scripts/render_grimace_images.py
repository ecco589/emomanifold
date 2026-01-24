#!/usr/bin/env python3
"""
使用grimaceface渲染四句语料，输出每个token的表情图片
"""
from pathlib import Path
from typing import List, Dict
import json
import math
import sys
import urllib.request

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from src.model_loader import ModelLoader
from src.dimension_extractor import DimensionExtractor
from src.emotion_projector import EmotionProjector
from src.adaptive_scaler import AdaptiveScaler
from src.fkcn import FKCN


TEXTS = [
    "我今天非常开心，心情特别好！",
    "听到这个消息，我感到非常失望和沮丧。",
    "这个结果让我既惊讶又兴奋。",
    "我现在很平静，感觉很放松。",
]


def _smooth_weights(weights: List[List[float]], alpha: float = 0.6) -> List[List[float]]:
    if not weights:
        return []
    smoothed = [weights[0][:]]
    for idx in range(1, len(weights)):
        prev = smoothed[-1]
        cur = weights[idx]
        blended = [(alpha * c + (1 - alpha) * p) for p, c in zip(prev, cur)]
        smoothed.append(blended)
    return smoothed


def _fkcn_to_grimace_weights(fkcn_weights: List[List[float]]) -> List[List[float]]:
    # FKCN顺序: neutral, happiness, surprise, fear, sadness, disgust, anger
    # grimace顺序: joy, surprise, fear, sadness, disgust, anger
    grimace = []
    for w in fkcn_weights:
        if len(w) < 7:
            raise ValueError("FKCN权重维度不足")
        grimace.append([w[1], w[2], w[3], w[4], w[5], w[6]])
    return grimace


def build_sequences(model_name: str) -> List[Dict]:
    model_config = config.ModelConfig(model_name)
    if not model_config.is_available():
        raise FileNotFoundError(f"模型 {model_name} 不可用: {model_config.local_path}")

    original_model = config.CURRENT_MODEL
    config.CURRENT_MODEL = model_name
    config.MODEL_NAME = model_config.model_id
    config.MODEL_LOCAL_PATH = model_config.local_path

    try:
        model_loader = ModelLoader()
        model_loader.load()

        dimension_extractor = DimensionExtractor(
            model_loader.get_output_embedding_matrix(),
            model_loader.tokenizer
        )
        vad_lexicon = dimension_extractor.load_vad_lexicon()
        dimensions = dimension_extractor.extract_all_dimensions(vad_lexicon)

        scaler_path = project_root / "outputs" / "scalers" / f"{model_name}_scaler.json"
        scaler = AdaptiveScaler(vad_lexicon)
        scaler.load(str(scaler_path))

        projector = EmotionProjector(dimensions, scaler=scaler)
        fkcn = FKCN(device=model_loader.device)

        sequences = []
        for text in TEXTS:
            emotion_coords, tokens = projector.project_text(model_loader, text, apply_scaling=True)
            valence = emotion_coords.get("valence", [])
            arousal = emotion_coords.get("arousal", [])
            points = list(zip(valence, arousal))
            fkcn_weights = fkcn.compute_weights_batch(points)

            grimace_weights = _fkcn_to_grimace_weights(fkcn_weights)
            grimace_weights = _smooth_weights(grimace_weights, alpha=0.6)

            sequences.append({
                "text": text,
                "tokens": tokens,
                "weights": grimace_weights
            })
        return sequences
    finally:
        config.CURRENT_MODEL = original_model


def _build_html(sequence: Dict, width: int, height: int) -> str:
    data = json.dumps(sequence, ensure_ascii=False)
    return f"""
<!DOCTYPE html>
<html lang="zh">
  <head>
    <meta charset="UTF-8" />
    <title>Grimace Render</title>
    <style>
      body {{
        margin: 0;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        font-family: Arial, sans-serif;
      }}
      #grimace-container {{
        width: {width}px;
        height: {width}px;
      }}
      #subtitle {{
        margin-top: 16px;
        font-size: 28px;
        color: #111111;
        height: 40px;
      }}
    </style>
  </head>
  <body>
    <div id="grimace-container"></div>
    <div id="subtitle"></div>
    <script>
      const sequence = {data};
      const container = document.getElementById('grimace-container');
      const subtitle = document.getElementById('subtitle');

      window.grimaceInstance = null;
      window.initGrimace = () => {{
        // 清理之前的实例
        if (window.grimaceInstance) {{
          try {{
            const canvas = container.querySelector('canvas');
            if (canvas) canvas.remove();
          }} catch(e) {{}}
        }}
        const GrimaceCtor = window.GrimaceModule?.default || window.GrimaceModule;
        window.grimaceInstance = new GrimaceCtor(container);

        function setIndex(index) {{
          const weights = sequence.weights[index] || [0,0,0,0,0,0];
          // 放大权重效果，使表情更明显
          // 方法：找到最大权重，然后按比例放大所有权重，确保最大权重接近1.0
          const maxWeight = Math.max(...weights, 0.01);
          const scale = maxWeight > 0 ? Math.min(1.0 / maxWeight, 2.0) : 1.0;
          const amplified = weights.map(w => Math.min(1.0, w * scale));
          window.grimaceInstance.setEmotionSet({{
            joy: amplified[0],
            surprise: amplified[1],
            fear: amplified[2],
            sadness: amplified[3],
            disgust: amplified[4],
            anger: amplified[5],
          }}, 0);
          subtitle.textContent = sequence.tokens[index] || '';
        }}

        window.setIndex = setIndex;
        window.getTokenCount = () => sequence.tokens.length;
      }};
    </script>
  </body>
</html>
"""


def _load_grimace_bundle(cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://unpkg.com/grimace/dist/index.js"
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    cache_path.write_text(content, encoding="utf-8")
    return content


def render_images(sequences: List[Dict], output_dir: Path):
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    grimace_js = _load_grimace_bundle(output_dir / "grimace.umd.js")

    width = 600
    height = 740

    with sync_playwright() as p:
        browser = p.chromium.launch()

        for idx, sequence in enumerate(sequences, start=1):
            # 为每个句子创建新页面，确保完全隔离
            page = browser.new_page(viewport={"width": width, "height": height})
            
            html = _build_html(sequence, width=width, height=height)
            page.set_content(html, wait_until="domcontentloaded")
            
            # 先设置require和module环境
            page.evaluate("""
                window.module = { exports: {} };
                window.exports = window.module.exports;
                window.require = (name) => {
                  if (name && name !== 'ts-deepmerge') {
                    return {};
                  }
                  const isObject = (val) => val && typeof val === 'object' && !Array.isArray(val);
                  const merge = (target, source) => {
                    if (Array.isArray(target) && Array.isArray(source)) {
                      return target.concat(source);
                    }
                    if (!isObject(target) || !isObject(source)) {
                      return source;
                    }
                    const out = { ...target };
                    Object.keys(source).forEach((key) => {
                      if (key in out) {
                        out[key] = merge(out[key], source[key]);
                      } else {
                        out[key] = source[key];
                      }
                    });
                    return out;
                  };
                  return (...items) => items.reduce((acc, item) => merge(acc, item), {});
                };
            """)
            # 加载grimace脚本
            page.add_script_tag(content=grimace_js)
            # 等待脚本执行完成
            page.wait_for_timeout(800)  # 等待脚本加载和执行
            page.evaluate("window.GrimaceModule = window.module.exports;")
            page.wait_for_function("typeof window.GrimaceModule !== 'undefined'", timeout=15000)
            # 初始化grimace
            page.evaluate("initGrimace()")
            page.wait_for_function("window.getTokenCount !== undefined && window.setIndex !== undefined", timeout=10000)
            # 等待初始渲染完成（canvas绘制需要时间）
            page.wait_for_timeout(400)

            token_count = page.evaluate("getTokenCount()")
            sentence_dir = output_dir / f"sentence_{idx:02d}"
            sentence_dir.mkdir(parents=True, exist_ok=True)

            for t in range(token_count):
                page.evaluate(f"setIndex({t})")
                # 等待grimace渲染完成（canvas需要时间绘制）
                page.wait_for_timeout(400)  # 等待确保渲染完成
                frame_path = sentence_dir / f"token_{t:03d}.png"
                page.screenshot(path=str(frame_path))
            
            # 关闭当前页面，释放资源
            page.close()

        browser.close()


def main():
    output_dir = project_root / "outputs" / "grimace_images"
    sequences = build_sequences(model_name="ernie")
    render_images(sequences, output_dir=output_dir)
    print(f"已生成图片: {output_dir}")


if __name__ == "__main__":
    main()
