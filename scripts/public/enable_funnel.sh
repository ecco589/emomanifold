#!/usr/bin/env bash
# Public HTTPS for the lab model server (free, no upload).
# 1. Open the URL printed by `tailscale funnel` and enable Funnel for your tailnet.
# 2. Re-run this script, then set GitHub repo variable VITE_MODEL_HOST to the funnel URL.
set -euo pipefail
LAB="${EMO_LAB_SSH:-nexgen-lab}"

ssh -t "$LAB" bash -s <<'REMOTE'
set -euo pipefail
if ! pgrep -f "serve_models.py" >/dev/null; then
  nohup python3 ~/emomanifold/bin/serve_models.py > ~/emomanifold/model-server.log 2>&1 &
  sleep 1
fi
tailscale funnel --bg 8787 2>&1 || true
echo ""
tailscale funnel status
echo ""
echo "Set GitHub → Settings → Variables → VITE_MODEL_HOST to the https://… URL above."
REMOTE
