#!/usr/bin/env bash
# 用 hf-mirror + curl 预下载 Py-Feat 权重到 data/pyfeat_models/
# 避开卡住的 huggingface_hub
set -e
# RAL/experiment/C1/code/mirror → 仓库根目录（上 5 级）
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
OUT="$ROOT/data/pyfeat_models"
MIRROR="https://hf-mirror.com"
mkdir -p "$OUT/retinaface_r34" "$OUT/face_multitask_v2"

echo "[1/2] retinaface_r34 (≈85MB)"
curl -L -C - --connect-timeout 10 \
  -o "$OUT/retinaface_r34/model.safetensors" \
  "$MIRROR/py-feat/retinaface_r34/resolve/main/model.safetensors" \
  --progress-bar
ls -lh "$OUT/retinaface_r34/model.safetensors"

echo "[2/2] face_multitask_v2 (≈146MB)"
curl -L -C - --connect-timeout 10 \
  -o "$OUT/face_multitask_v2/face_multitask_v2.safetensors" \
  "$MIRROR/py-feat/face_multitask_v2/resolve/main/face_multitask_v2.safetensors" \
  --progress-bar
ls -lh "$OUT/face_multitask_v2/face_multitask_v2.safetensors"
echo "完成: $OUT"
