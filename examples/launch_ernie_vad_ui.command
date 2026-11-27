#!/bin/bash
# macOS 可双击。从仓库根目录启动 Gradio 流式界面。
cd "$(dirname "$0")/.."
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GRADIO_ANALYTICS_ENABLED=false
if [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then
  PY="./venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -u examples/stream_ernie_vad_ui.py
