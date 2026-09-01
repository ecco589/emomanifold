#!/usr/bin/env bash
set -euo pipefail
LAB="${EMO_LAB_SSH:-nexgen-lab}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST_SUFFIX="onnx-community/Qwen2.5-0.5B-Instruct/resolve/main"

ssh "$LAB" "mkdir -p ~/emomanifold/bin"
scp "$REPO_ROOT/scripts/lab/serve_models.py" "$LAB:~/emomanifold/bin/serve_models.py"

ssh "$LAB" bash -s <<REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
pip install --user -q huggingface_hub
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
DEST="\$HOME/emomanifold-models/$DEST_SUFFIX"
mkdir -p "\$DEST/onnx"

if [ ! -f "\$DEST/tokenizer_config.json" ]; then
  hf download onnx-community/Qwen2.5-0.5B-Instruct \\
    --local-dir "\$DEST" \\
    --include "*.json" "merges.txt" "vocab.json"
fi

if [ ! -f "\$DEST/onnx/model_q4.onnx" ]; then
  echo "Downloading ONNX (~300MB)…"
  hf download onnx-community/Qwen2.5-0.5B-Instruct onnx/model_q4.onnx --local-dir "\$DEST"
fi

ls -lh "\$DEST/onnx/model_q4.onnx"
pkill -f "serve_models.py" 2>/dev/null || true
nohup python3 ~/emomanifold/bin/serve_models.py > ~/emomanifold/model-server.log 2>&1 &
sleep 1
curl -sfI "http://127.0.0.1:8787/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json" | head -3
REMOTE

echo "OK — http://ubuntu.tail1701c2.ts.net:8787/"
