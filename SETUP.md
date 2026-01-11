# 环境设置指南

## 为什么使用虚拟环境？

虚拟环境可以：
- 隔离项目依赖，不影响系统全局 Python 环境
- 避免不同项目的依赖冲突
- 保持项目环境的可重现性

## 设置步骤

### 1. 创建虚拟环境

在项目根目录下运行：

```bash
cd /Users/eccoliu/Desktop/EmotionManifold
python3 -m venv venv
```

这会在项目目录下创建一个 `venv` 文件夹（已添加到 `.gitignore`，不会被提交到 Git）。

### 2. 激活虚拟环境

**macOS/Linux:**
```bash
source venv/bin/activate
```

激活后，你的终端提示符前会显示 `(venv)`，表示虚拟环境已激活。

### 3. 安装项目依赖

在虚拟环境激活状态下：

```bash
pip install -r requirements.txt
```

或者使用安装脚本：
```bash
bash scripts/install_dependencies.sh
```

### 4. 验证安装

检查关键包是否安装成功：

```bash
pip list | grep -E "(torch|transformers|modelscope|numpy|matplotlib)"
```

### 5. 退出虚拟环境（可选）

当你完成工作后，可以退出虚拟环境：

```bash
deactivate
```

## 后续使用

每次开始工作时，记得先激活虚拟环境：

```bash
cd /Users/eccoliu/Desktop/EmotionManifold
source venv/bin/activate
```

然后就可以运行项目代码了。

## 注意事项

- 虚拟环境文件夹 `venv/` 已添加到 `.gitignore`，不会被提交到 Git
- 每个开发者需要在自己的机器上创建虚拟环境
- 如果遇到权限问题，确保你在项目目录下有写入权限
