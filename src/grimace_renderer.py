"""
Grimace 单帧渲染器：给定 6 维表情权重，渲染为 PNG 图像
供 stream_ernie_vad_ui、render_grimace_images 复用
"""
import json
import urllib.request
from pathlib import Path
from typing import List, Optional

# 项目根用于缓存
_project_root = Path(__file__).parent.parent


def load_grimace_bundle(cache_path: Optional[Path] = None) -> str:
    """加载 Grimace UMD 脚本（缓存到本地）"""
    if cache_path is None:
        cache_path = _project_root / "outputs" / "grimace_images" / "grimace.umd.js"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://unpkg.com/grimace/dist/index.js"
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    cache_path.write_text(content, encoding="utf-8")
    return content


PLAYWRIGHT_INIT_JS = """
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
"""


def _build_streaming_html(width: int, height: int) -> str:
    """生成支持 setWeights(weights) 的流式渲染 HTML"""
    return f"""
<!DOCTYPE html>
<html lang="zh">
  <head>
    <meta charset="UTF-8" />
    <title>Grimace</title>
    <style>
      body {{ margin: 0; background: #fff; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: Arial,sans-serif; }}
      #grimace-container {{ width: {width}px; height: {width}px; }}
    </style>
  </head>
  <body>
    <div id="grimace-container"></div>
    <script>
      const container = document.getElementById('grimace-container');
      window.grimaceInstance = null;
      window.initGrimace = () => {{
        if (window.grimaceInstance) {{
          try {{ container.querySelector('canvas')?.remove(); }} catch(e) {{}}
        }}
        const GrimaceCtor = window.GrimaceModule?.default || window.GrimaceModule;
        window.grimaceInstance = new GrimaceCtor(container);
        window.setWeights = (weights) => {{
          const w = weights || [0,0,0,0,0,0];
          const maxWeight = Math.max(...w, 0.01);
          const scale = maxWeight > 0 ? Math.min(1.0 / maxWeight, 2.0) : 1.0;
          const amplified = w.map(x => Math.min(1.0, x * scale));
          window.grimaceInstance.setEmotionSet({{
            joy: amplified[0],
            surprise: amplified[1],
            fear: amplified[2],
            sadness: amplified[3],
            disgust: amplified[4],
            anger: amplified[5],
          }}, 0);
        }};
      }};
    </script>
  </body>
</html>
"""


class GrimaceFrameRenderer:
    """
    单帧 Grimace 渲染器，维护一个 Playwright 页面用于流式渲染。
    使用方式: with GrimaceFrameRenderer(...) as r: r.render(weights) -> bytes
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 500,
        cache_dir: Optional[Path] = None,
    ):
        self.width = width
        self.height = height
        self._cache_dir = cache_dir or (_project_root / "outputs" / "grimace_images")
        self._browser = None
        self._page = None
        self._grimace_js: Optional[str] = None

    def __enter__(self) -> "GrimaceFrameRenderer":
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(args=[
            "--disable-background-networking",
            "--disable-client-side-phishing-detection",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--no-first-run",
        ])
        self._page = self._browser.new_page(viewport={"width": self.width, "height": self.height})

        html = _build_streaming_html(self.width, self.height)
        self._page.set_content(html, wait_until="domcontentloaded")
        self._page.evaluate(PLAYWRIGHT_INIT_JS)
        js_path = self._cache_dir / "grimace.umd.js"
        self._grimace_js = load_grimace_bundle(js_path)
        self._page.add_script_tag(content=self._grimace_js)
        self._page.wait_for_timeout(600)
        self._page.evaluate("window.GrimaceModule = window.module.exports;")
        self._page.wait_for_function("typeof window.GrimaceModule !== 'undefined'", timeout=10000)
        self._page.evaluate("initGrimace()")
        self._page.wait_for_function("typeof window.setWeights === 'function'", timeout=5000)
        self._page.wait_for_timeout(300)
        return self

    def __exit__(self, *args) -> None:
        if self._page:
            self._page.close()
        if self._browser:
            self._browser.close()
        if hasattr(self, "_playwright"):
            self._playwright.stop()

    def render(self, weights: List[float]) -> bytes:
        """渲染单帧，返回 PNG bytes。weights 为 6 维 [joy, surprise, fear, sadness, disgust, anger]"""
        if len(weights) < 6:
            weights = (weights + [0.0] * 6)[:6]
        w_json = json.dumps(weights)
        self._page.evaluate(f"setWeights({w_json})")
        self._page.wait_for_timeout(200)
        return self._page.screenshot(type="png")
