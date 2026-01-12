#!/bin/bash
# DeepSeek 模型交互式对话启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 运行对话脚本
python3 chat.py

# 如果脚本出错，保持终端打开以便查看错误信息
if [ $? -ne 0 ]; then
    echo ""
    echo "程序异常退出，按任意键关闭..."
    read -n 1
fi
