#!/bin/bash
# 依赖安装脚本
# 使用方法: bash scripts/install_dependencies.sh

echo "正在安装 EmotionManifold 项目依赖..."

# 安装 transformers
echo "安装 transformers..."
pip3 install transformers>=4.35.0

# 安装 modelscope
echo "安装 modelscope..."
pip3 install modelscope>=1.9.0

# 安装其他依赖
echo "安装其他依赖..."
pip3 install pandas>=2.0.0 seaborn>=0.12.0 tqdm>=4.65.0 pyyaml>=6.0

echo "依赖安装完成！"
echo ""
echo "已安装的包："
pip3 list | grep -E "(torch|transformers|modelscope|numpy|matplotlib|pandas|seaborn|tqdm|pyyaml)"
