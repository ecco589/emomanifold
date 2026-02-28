#!/bin/bash
cd "$(dirname "$0")"
export TRANSFORMERS_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=false
exec /Users/eccoliu/miniforge3/bin/python -u examples/stream_ernie_vad_ui.py
