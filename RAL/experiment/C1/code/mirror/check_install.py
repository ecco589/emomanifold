from __future__ import annotations

import importlib
import shutil
import sys


PACKAGES = [
    "cv2",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "torch",
]


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    failed: list[str] = []

    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"OK  {package}: {version}")
        except Exception as exc:
            failed.append(package)
            print(f"ERR {package}: {exc}")

    if failed:
        print("")
        print("Missing packages:", ", ".join(failed))
        print("Run setup_env.bat again.")
        return 1

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print("OK  ffmpeg:", ffmpeg_path)
    else:
        print("WARN ffmpeg: not found on PATH")
        print("     TorchCodec on Windows needs a shared FFmpeg build.")

    try:
        import torchcodec  # noqa: F401

        print("OK  torchcodec: can load shared libraries")
    except Exception as exc:
        message = str(exc).lower()
        if "ffmpeg" in message or "libtorchcodec" in message:
            print("ERR torchcodec: could not load FFmpeg shared libraries")
            print("    Run install_ffmpeg_hint.bat, then open a new window and retry.")
        else:
            print("ERR torchcodec:", exc)
        return 1

    try:
        from feat import Detectorv2  # noqa: F401

        print("OK  feat.Detectorv2 is available")
    except Exception as exc:
        print("ERR Detectorv2 is not available:", exc)
        print("Try running setup_env.bat again.")
        return 1

    print("")
    print("Install check passed.")
    print("The first real run may still take time because Py-Feat downloads model weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
