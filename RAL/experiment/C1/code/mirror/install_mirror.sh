#!/usr/bin/env bash
# C1 Mirror 依赖安装（清华镜像 + 进度）
set -e
cd "$(dirname "$0")"
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
PY="${PYTHON:-python3.12}"
command -v "$PY" &>/dev/null || PY=python3

step() { echo ""; echo ">>> [$1] $2"; echo ">>> $(date '+%H:%M:%S')"; }

step "1/5" "Python $($PY --version 2>&1)"
[ -d .venv ] && rm -rf .venv
"$PY" -m venv .venv
source .venv/bin/activate
pip install $MIRROR --upgrade pip

step "2/5" "numpy pandas matplotlib opencv openpyxl tqdm"
pip install $MIRROR numpy pandas matplotlib openpyxl opencv-python-headless tqdm

step "3/5" "torch CPU（约 200MB）"
pip install $MIRROR torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
  pip install $MIRROR torch torchvision

step "4/5" "py-feat"
pip install $MIRROR py-feat

step "5/5" "验证"
python -c "from feat import Detectorv2; print('Detectorv2 OK')"
echo ""
echo "=== 安装完成 $(date '+%H:%M:%S') ==="
