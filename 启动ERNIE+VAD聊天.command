#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate && python examples/stream_ernie_vad_ui.py
