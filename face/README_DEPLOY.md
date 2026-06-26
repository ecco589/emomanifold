# Face M3 部署说明（VA → 实机表情）

最小闭环：网页输入 VA → FKCN 7 权重 → 融合 7 套表情 factor → 29 路舵机。

## 文件
- `face_driver.py`  核心：FKCN + 读 7 套 yaml 融合 factor + 下发（无串口自动 dry-run）
- `server.py`       极简 HTTP 后端，托管 webui + `POST /apply`
- `webui/`          网页界面（VA 图 + Robot 模式）
- `Motor.py / Communication.py / utility.py`  原厂例程（未改动）
- `29_servo_config.yaml`、`expression29/*.yaml`  舵机配置与 7 套表情（未改动）

## 工控机（Linux）运行

与 M4 共用仓库根目录 `venv`（勿在 `face/` 下另建环境）：

```bash
cd ..                    # 仓库根目录
source venv/bin/activate
pip install -r requirements.txt   # 首次（含 pyserial）
cd face
# 如串口需要权限：sudo chmod 777 /dev/usb_face
python -u server.py
```

M4 整仓 UI：`python examples/stream_ernie_vad_ui.py`（同一 venv，勾选「实机」）。
浏览器打开 `http://127.0.0.1:8770/`，勾选「Robot mode」后改 VA 即下发实机。

## 配置（face_driver.py 顶部）
- `SERIAL_PORT`  默认 `/dev/usb_face`
- `BAUDRATE`     默认 9600
- `MOVE_TIME_MS` 每次动作时长（建议 ≥100ms）
- `AUTO_CHMOD`   置 True 则启动时自动 `sudo chmod 777` 串口

## 模式说明
- **dry-run**：非 Linux 或串口打不开时自动进入，只在终端打印 29 路 factor，不下发。
- **live**：Linux + 串口可用时自动下发舵机。
- 也可在创建时强制：`FaceDriver(dry_run=True/False)`。

## 命令行自测（不依赖网页）
```bash
python -u face_driver.py     # 打印若干 VA 样本的 7 权重与 29 factor
```

## 注意
- FKCN 原型与项目 `src/fkcn.py` 一致；网页前端仅作预览，下发以 Python 端为准。
- 表情映射：happiness→happy, sadness→sad, anger→angry。
- factor 融合天然落在 [0,1]，不会超出舵机范围。
