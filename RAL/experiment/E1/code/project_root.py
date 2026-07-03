"""定位仓库根目录（含 config.py），与 experiment 所在层级无关。"""
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "config.py").is_file():
            return p
    raise FileNotFoundError("Cannot locate project root (config.py)")
