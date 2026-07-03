#!/usr/bin/env python3
"""M4: 流式 ERNIE + VAD + 表情（10fps）+ 可选实机 29 DOF"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
print("正在启动...", flush=True)

import io
import threading
import warnings
warnings.filterwarnings("ignore", module="urllib3")
from pathlib import Path
import html

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("加载依赖库...", flush=True)
import numpy as np
from PIL import Image

FACE_FPS = 10.0

_model_ready = threading.Event()
_chat_fn = None
_robot_sink = None

history = []


def _bytes_to_image(img_bytes: bytes):
    return np.array(Image.open(io.BytesIO(img_bytes)))


_face_worker = None
_face_worker_lock = threading.Lock()


def _get_face_worker():
    global _face_worker
    with _face_worker_lock:
        if _face_worker is not None and _face_worker.ok:
            return _face_worker
        from src.grimace_renderer import get_render_worker
        _face_worker = get_render_worker(width=400, height=450)
        return _face_worker


def _render_face(worker, weights):
    if worker is None:
        return None
    return worker.render(weights)


TOOLTIP_CSS = """
<style>
.vad{position:relative;cursor:help;border-bottom:1px dotted #999}
.vad:hover::after{content:attr(data-vad);position:absolute;left:0;bottom:100%;background:#333;color:#fff;padding:4px 8px;font-size:12px;white-space:nowrap;z-index:9999;border-radius:4px;margin-bottom:4px}
.round{margin-bottom:0.8em}
</style>
"""


def to_html(msg, tokens, model_label="Qwen"):
    parts = [f"<div class='round'><b>你:</b> {html.escape(msg)}</div><div class='round'><b>{html.escape(model_label)}:</b> "]
    for t, v in tokens:
        vv, aa = v.get("valence", 0), v.get("arousal", 0)
        dd = v.get("dominance")
        if dd is not None:
            tip = f"V:{vv:.2f} A:{aa:.2f} D:{dd:.2f}"
        else:
            tip = f"V:{vv:.2f} A:{aa:.2f}"
        parts.append(f'<span class="vad" data-vad="{tip}">{html.escape(t)}</span>')
    parts.append("</div>")
    return TOOLTIP_CSS + "".join(parts)


def make_chat_fn(model, tokenizer, projector, fkcn, device, robot_sink, model_label="Qwen"):
    from src.stream_engine import generate_with_face_stream
    from src.face_interpolator import FaceFrameInterpolator
    from src.grimace_utils import fkcn_to_grimace_weights

    def chat(msg):
        if not msg.strip():
            yield "", None, robot_sink.status
            return
        tokens = []
        worker = _get_face_worker()
        interpolator = FaceFrameInterpolator(
            fps=FACE_FPS,
            n_dims=7,
            realtime=True,
            max_settle_frames=2,
        )
        messages = [
            {"role": "system", "content": "请简洁回复，一两句话即可。不要啰嗦，少用表情符号。"},
        ]
        for m, ts in history:
            messages.append({"role": "user", "content": m})
            messages.append({"role": "assistant", "content": "".join(t for t, _ in ts)})
        messages.append({"role": "user", "content": msg})
        html_out = ""
        for t, v, weights7 in generate_with_face_stream(
            model, tokenizer, projector, messages,
            fkcn, device=device,
            max_new_tokens=96,
            temperature=0.4,
        ):
            tokens.append((t, v))
            html_out = "".join(to_html(m, ts, model_label) for m, ts in history + [(msg, tokens)])
            for frame_w7 in interpolator.animate_to(weights7):
                grimace_6 = fkcn_to_grimace_weights(frame_w7)
                img_arr = None
                if worker is not None:
                    img_bytes = _render_face(worker, grimace_6)
                    if img_bytes:
                        img_arr = _bytes_to_image(img_bytes)
                robot_sink.apply_weights7(frame_w7)
                yield html_out, img_arr, robot_sink.status
        history.append((msg, tokens))

    return chat


def _load_model_in_background(robot_sink):
    global _chat_fn
    try:
        import config
        model_name = config.get_active_model()
        print(f"加载模型 {model_name}（约 30 秒）...", flush=True)
        from src.stream_engine import load_chat_model_and_projector, load_fkcn
        model, tokenizer, projector = load_chat_model_and_projector(model_name)
        fkcn = load_fkcn(device="cpu")
        device = str(next(model.parameters()).device)
        model_label = model_name.upper() if model_name == "ernie" else model_name.capitalize()
        _chat_fn = make_chat_fn(model, tokenizer, projector, fkcn, device, robot_sink, model_label)
        _model_ready.set()
        print("模型就绪", flush=True)
    except Exception as e:
        print(f"加载失败: {e}", flush=True)


def chat_wrapper(msg):
    if not msg.strip():
        yield "", None, _robot_sink.status if _robot_sink else "实机: 关"
        return
    _model_ready.wait(timeout=300)
    if _chat_fn is None:
        yield "模型加载失败，请刷新重试", None, _robot_sink.status if _robot_sink else ""
        return
    for h, img, st in _chat_fn(msg):
        yield h, img, st


def on_robot_toggle(enabled):
    return _robot_sink.set_enabled(enabled)


if __name__ == "__main__":
    import os
    import time

    from src.robot_face_sink import RobotFaceSink

    print("加载 Gradio...", flush=True)
    import gradio as gr
    print("准备界面...", flush=True)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    _robot_sink = RobotFaceSink()

    try:
        import gradio.networking as net
        net.url_ok = lambda url: True
    except Exception:
        pass

    def _delayed_load():
        time.sleep(2)
        threading.Thread(target=_get_face_worker, daemon=True).start()
        time.sleep(3)
        _load_model_in_background(_robot_sink)

    threading.Thread(target=_delayed_load, daemon=True).start()

    import config
    active_model = config.get_active_model()
    model_title = active_model.upper() if active_model == "ernie" else active_model.capitalize()
    vad_dims = "VAD" if active_model == "qwen" else "VA"
    with gr.Blocks(title=f"{model_title} + {vad_dims} + 表情 + 实机") as demo:
        gr.Markdown(
            f"## {model_title} 流式对话 · 悬停 token 查看 {vad_dims} · "
            f"右侧逐 token 表情（{int(FACE_FPS)}fps，与实机同步）"
        )
        gr.Markdown("*模型加载中，就绪后可发送消息*")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(placeholder="输入消息，按回车或点击发送", label="消息", lines=2)
                with gr.Row():
                    btn = gr.Button("发送", variant="primary")
                    robot_cb = gr.Checkbox(label="实机", value=False)
                robot_status = gr.Textbox(label="实机状态", value="实机: 关", interactive=False)
                chat_out = gr.HTML(label="对话")
            with gr.Column(scale=1):
                face_out = gr.Image(label="表情", height=400)

        def on_load():
            global history
            history.clear()
            return ""

        robot_cb.change(on_robot_toggle, inputs=[robot_cb], outputs=[robot_status])
        inp.submit(chat_wrapper, inputs=[inp], outputs=[chat_out, face_out, robot_status])
        btn.click(chat_wrapper, inputs=[inp], outputs=[chat_out, face_out, robot_status])
        demo.load(on_load, outputs=[chat_out])

    print(f"请打开 http://127.0.0.1:{port}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=False)
