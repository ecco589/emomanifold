"""Render Grimace 6-D weights to PNG frames and MP4.

The browser bundle is vendored. No network access at render time.
Playwright is imported only when a renderer is opened.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

_VENDOR = Path(__file__).resolve().parent / "vendor" / "grimace.umd.js"

_PLAYWRIGHT_INIT_JS = """
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


def _html(width: int, height: int, amplify: bool) -> str:
    amplify_js = "true" if amplify else "false"
    return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Grimace</title>
    <style>
      body {{
        margin: 0;
        background: #fff;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        font-family: Arial, sans-serif;
      }}
      #grimace-container {{ width: {width}px; height: {width}px; }}
    </style>
  </head>
  <body>
    <div id="grimace-container"></div>
    <script>
      const AMPLIFY = {amplify_js};
      const container = document.getElementById('grimace-container');
      window.grimaceInstance = null;
      window.initGrimace = () => {{
        if (window.grimaceInstance) {{
          try {{ container.querySelector('canvas')?.remove(); }} catch (e) {{}}
        }}
        const GrimaceCtor = window.GrimaceModule?.default || window.GrimaceModule;
        window.grimaceInstance = new GrimaceCtor(container);
        window.setWeights = (weights) => {{
          let w = weights || [0, 0, 0, 0, 0, 0];
          if (AMPLIFY) {{
            const maxWeight = Math.max(...w, 0.01);
            const scale = maxWeight > 0 ? Math.min(1.0 / maxWeight, 2.0) : 1.0;
            w = w.map((x) => Math.min(1.0, x * scale));
          }} else {{
            w = w.map((x) => Math.max(0, Math.min(1, x)));
          }}
          window.grimaceInstance.setEmotionSet({{
            joy: w[0],
            surprise: w[1],
            fear: w[2],
            sadness: w[3],
            disgust: w[4],
            anger: w[5],
          }}, 0);
        }};
      }};
    </script>
  </body>
</html>
"""


class GrimaceRenderer:
    """One Chromium page. Call ``render`` from the thread that opened it.

    Parameters
    ----------
    width, height:
        Viewport in pixels. The face is square, sized to ``width``.
    amplify:
        Rescale each frame so its strongest weight sits near 1, at most 2×.
        This is what the experiment videos do. Set False to draw the raw
        FKCN mixture.
    frame_wait_ms:
        Pause after setting weights so the canvas has painted.
    bundle_path:
        Override the vendored ``grimace.umd.js``.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        *,
        amplify: bool = True,
        frame_wait_ms: int = 80,
        bundle_path: str | Path | None = None,
    ):
        self.width = width
        self.height = height
        self.amplify = amplify
        self.frame_wait_ms = frame_wait_ms
        self.bundle_path = Path(bundle_path) if bundle_path else _VENDOR
        self._playwright = None
        self._browser = None
        self._page = None

    def open(self) -> "GrimaceRenderer":
        if self._page is not None:
            return self
        if not self.bundle_path.is_file():
            raise FileNotFoundError(f"Grimace bundle not found: {self.bundle_path}")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "playwright is not installed. Run: pip install playwright "
                "&& python -m playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            args=[
                "--disable-background-networking",
                "--disable-client-side-phishing-detection",
                "--disable-default-apps",
                "--disable-extensions",
                "--disable-sync",
                "--no-first-run",
            ]
        )
        self._page = self._browser.new_page(
            viewport={"width": self.width, "height": self.height}
        )
        self._page.set_content(
            _html(self.width, self.height, self.amplify),
            wait_until="domcontentloaded",
        )
        self._page.evaluate(_PLAYWRIGHT_INIT_JS)
        self._page.add_script_tag(content=self.bundle_path.read_text(encoding="utf-8"))
        self._page.wait_for_timeout(600)
        self._page.evaluate("window.GrimaceModule = window.module.exports;")
        self._page.wait_for_function(
            "typeof window.GrimaceModule !== 'undefined'", timeout=10000
        )
        self._page.evaluate("initGrimace()")
        self._page.wait_for_function(
            "typeof window.setWeights === 'function'", timeout=5000
        )
        self._page.wait_for_timeout(300)
        # Paint a neutral face once so the first real expression is not a cold start.
        self.render([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self

    def close(self) -> None:
        if self._page is not None:
            self._page.close()
            self._page = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __enter__(self) -> "GrimaceRenderer":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    def render(self, weights: Sequence[float]) -> bytes:
        """Return one PNG. ``weights`` is 6-D Grimace order."""
        if self._page is None:
            self.open()
        w = [float(x) for x in weights]
        if len(w) < 6:
            w = (w + [0.0] * 6)[:6]
        elif len(w) > 6:
            w = w[:6]
        self._page.evaluate(f"setWeights({json.dumps(w)})")
        self._page.wait_for_timeout(self.frame_wait_ms)
        return self._page.screenshot(type="png")

    def save(self, weights: Sequence[float], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.render(weights))
        return path


def frames_to_mp4(pngs: Sequence[bytes], out_mp4: str | Path, fps: float) -> Path:
    """Encode PNG frames with ffmpeg (libx264, yuv420p)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grimace_frames_") as tmp:
        tmp_dir = Path(tmp)
        for i, png in enumerate(pngs):
            (tmp_dir / f"frame_{i:05d}.png").write_bytes(png)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise RuntimeError(f"ffmpeg failed:\n{tail}")
    return out_mp4
