#!/usr/bin/env bash
# Run Cloudflare Tunnel connector for EmoManifold API (port 8788).
# Usage: EMO_CF_TOKEN='eyJ...' ./start_cloudflare_tunnel.sh
set -euo pipefail
TOKEN="${EMO_CF_TOKEN:?Set EMO_CF_TOKEN to the token from Cloudflare dashboard}"
BIN="${HOME}/bin/cloudflared"
if [ ! -x "$BIN" ]; then
  echo "Missing $BIN — run deploy_api.sh first or install cloudflared."
  exit 1
fi
pkill -f "cloudflared tunnel run" 2>/dev/null || true
nohup "$BIN" tunnel run --token "$TOKEN" > ~/emomanifold/cf-tunnel.log 2>&1 &
sleep 3
echo "Tunnel started. Log: ~/emomanifold/cf-tunnel.log"
tail -5 ~/emomanifold/cf-tunnel.log
