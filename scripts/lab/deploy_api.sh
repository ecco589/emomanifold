#!/usr/bin/env bash
set -euo pipefail
LAB="${EMO_LAB_SSH:-nexgen-lab}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RBASE="/home/user/EmotionManifold"

ssh "$LAB" "mkdir -p $RBASE/{src,scripts/lab,data,outputs/scalers,models/Qwen}"

rsync -az "$ROOT/src/" "$LAB:$RBASE/src/"
rsync -az "$ROOT/config.py" "$LAB:$RBASE/"
rsync -az "$ROOT/data/" "$LAB:$RBASE/data/"
rsync -az "$ROOT/outputs/scalers/" "$LAB:$RBASE/outputs/scalers/"
rsync -az "$ROOT/scripts/lab/" "$LAB:$RBASE/scripts/lab/"

ssh "$LAB" bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
pip install --user -q fastapi uvicorn huggingface_hub
MODEL="$HOME/EmotionManifold/models/Qwen/Qwen2.5-0.5B-Instruct"
if [ ! -f "$MODEL/config.json" ]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL"
fi
pkill -f "api_server.py" 2>/dev/null || true
cd "$HOME/EmotionManifold"
export EMO_DEVICE=cuda
nohup python3 scripts/lab/api_server.py > ~/emomanifold/api.log 2>&1 &
sleep 10
curl -sf http://127.0.0.1:8788/health && echo ""
REMOTE

echo "OK — http://ubuntu.tail1701c2.ts.net:8788"
