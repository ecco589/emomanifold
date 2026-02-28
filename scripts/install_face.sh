#!/bin/bash
# 安装表情渲染依赖（Playwright + Chromium）
# 首次使用表情功能前运行一次
cd "$(dirname "$0")/.."
if [ -d venv ]; then
    source venv/bin/activate
fi
pip install playwright -q
playwright install chromium
echo "表情依赖安装完成"
