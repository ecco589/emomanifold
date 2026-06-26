"""
实机表情下发：FKCN 7 维 → factor 融合 → FaceDriver。
M4 与 UI 同进程、同节拍（10fps），不另起 HTTP 服务。
"""
import sys
import threading
from pathlib import Path
from typing import List, Optional

_FACE_DIR = Path(__file__).resolve().parent.parent / "face"
if str(_FACE_DIR) not in sys.path:
    sys.path.insert(0, str(_FACE_DIR))

MOVE_TIME_MS = 100  # 与 10fps 插值对齐；硬件下限 ~100ms


class RobotFaceSink:
    """逐帧驱动 29 DOF；enabled=False 时 no-op。"""

    def __init__(self):
        self.enabled = False
        self._driver = None
        self._lock = threading.Lock()
        self._status = "实机: 关"

    @property
    def status(self) -> str:
        return self._status

    def set_enabled(self, enabled: bool) -> str:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._status = "实机: 关"
            return self._status
        try:
            driver = self._ensure_driver()
            mode = "dry-run" if driver.dry_run else "live"
            self._status = f"实机: 开 ({mode})"
        except Exception as e:
            self._status = f"实机: 错误 ({e})"
        return self._status

    def _ensure_driver(self):
        if self._driver is not None:
            return self._driver
        from face_driver import FaceDriver  # noqa: WPS433 — face/ on sys.path

        self._driver = FaceDriver()
        return self._driver

    def apply_weights7(self, weights7: List[float], time_ms: int = MOVE_TIME_MS) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                driver = self._ensure_driver()
                factors = driver.fuse(weights7)
                driver.apply_factors(factors, time_ms)
                mode = "dry-run" if driver.dry_run else "live"
                self._status = f"实机: 开 ({mode})"
            except Exception as e:
                self._status = f"实机: 错误 ({e})"
                print(f"[robot] {e}", flush=True)

    def reset(self, time_ms: int = MOVE_TIME_MS) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                driver = self._ensure_driver()
                driver.reset(time_ms)
            except Exception as e:
                self._status = f"实机: 错误 ({e})"
