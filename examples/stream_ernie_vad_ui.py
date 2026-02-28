#!/usr/bin/env python3
"""M1/M2: 流式 ERNIE + VAD + 表情，Gradio UI，悬停 token 看 VAD，右侧逐 token 表情"""
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

# 延迟导入：避免阻塞界面显示
_model_ready = threading.Event()
_chat_fn = None

history = []


def _bytes_to_image(img_bytes: bytes):
    """PNG bytes → numpy array for Gradio Image"""
    return np.array(Image.open(io.BytesIO(img_bytes)))


_face_renderer = None


def _get_face_renderer():
    """懒加载 Grimace 渲染器。若 playwright 未安装则返回 None，对话仍可用"""
    global _face_renderer
    if _face_renderer is not None:
        return _face_renderer
    try:
        from src.grimace_renderer import GrimaceFrameRenderer
        _face_renderer = GrimaceFrameRenderer(width=400, height=450)
        _face_renderer.__enter__()
        return _face_renderer
    except Exception:
        return None


TOOLTIP_CSS = """
<style>
.vad{position:relative;cursor:help;border-bottom:1px dotted #999}
.vad:hover::after{content:attr(data-vad);position:absolute;left:0;bottom:100%;background:#333;color:#fff;padding:4px 8px;font-size:12px;white-space:nowrap;z-index:9999;border-radius:4px;margin-bottom:4px}
.round{margin-bottom:0.8em}
</style>
"""


def to_html(msg, tokens):
    """tokens: [(t, vad), ...]，每轮对话独立成块"""
    parts = [f"<div class='round'><b>你:</b> {html.escape(msg)}</div><div class='round'><b>ERNIE:</b> "]
    for t, v in tokens:
        vv, aa = v.get("valence", 0), v.get("arousal", 0)
        parts.append(f'<span class="vad" data-vad="V:{vv:.2f} A:{aa:.2f}">{html.escape(t)}</span>')
    parts.append("</div>")
    return TOOLTIP_CSS + "".join(parts)


def make_chat_fn(model, tokenizer, projector, fkcn, device):
    """闭包：显式传入依赖，避免 Gradio worker 中全局变量未定义"""
    from examples.stream_ernie_vad import generate_with_face_stream

    def chat(msg):
        if not msg.strip():
            return "", None
        tokens = []
        renderer = _get_face_renderer()
        # 带历史记忆：将 history 转为多轮 messages
        messages = [
            {"role": "system", "content": "请简洁回复，一两句话即可。不要啰嗦，少用表情符号。"},
        ]
        for m, ts in history:
            messages.append({"role": "user", "content": m})
            messages.append({"role": "assistant", "content": "".join(t for t, _ in ts)})
        messages.append({"role": "user", "content": msg})
        for t, v, weights in generate_with_face_stream(
            model, tokenizer, projector, messages,
            fkcn, device=device,
            max_new_tokens=96,
            temperature=0.4,
        ):
            tokens.append((t, v))
            if renderer is not None:
                try:
                    img_bytes = renderer.render(weights)
                    img_arr = _bytes_to_image(img_bytes)
                except Exception:
                    img_arr = None
            else:
                img_arr = None
            yield "".join(to_html(m, ts) for m, ts in history + [(msg, tokens)]), img_arr
        history.append((msg, tokens))

    return chat


def _load_model_in_background():
    """后台加载模型，避免阻塞界面"""
    global _chat_fn
    try:
        print("加载模型（约 30 秒）...", flush=True)
        from examples.stream_ernie_vad import (
            load_chat_model_and_projector,
            load_fkcn,
        )
        model, tokenizer, projector = load_chat_model_and_projector("ernie")
        print("加载表情模块...", flush=True)
        fkcn = load_fkcn(device="cpu")
        device = str(next(model.parameters()).device)
        _chat_fn = make_chat_fn(model, tokenizer, projector, fkcn, device)
        _model_ready.set()
        print("模型就绪", flush=True)
    except Exception as e:
        print(f"加载失败: {e}", flush=True)


def chat_wrapper(msg):
    """等待模型就绪后调用实际 chat"""
    if not msg.strip():
        return "", None
    _model_ready.wait(timeout=300)
    if _chat_fn is None:
        yield "模型加载失败，请刷新重试", None
        return
    for h, img in _chat_fn(msg):
        yield h, img


if __name__ == "__main__":
    import os
    print("加载 Gradio...", flush=True)
    import gradio as gr
    print("准备界面...", flush=True)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    # 跳过 Gradio 启动时的 url 检测（慢机上易超时导致启动失败，服务器实际已启动）
    try:
        import gradio.networking as net
        net.url_ok = lambda url: True
    except Exception:
        pass

    # 界面先启动，模型延迟 5 秒后台加载（避免与服务器争抢资源）
    def _delayed_load():
        import time
        time.sleep(5)
        _load_model_in_background()
    t = threading.Thread(target=_delayed_load, daemon=True)
    t.start()

    with gr.Blocks(title="ERNIE + VAD + 表情") as demo:
        gr.Markdown("## ERNIE 流式对话 · 悬停 token 查看 VAD · 右侧逐 token 表情")
        gr.Markdown("*模型加载中，就绪后可发送消息*")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(placeholder="输入消息，按回车或点击发送", label="消息", lines=2)
                with gr.Row():
                    btn = gr.Button("发送", variant="primary")
                chat_out = gr.HTML(label="对话")
            with gr.Column(scale=1):
                face_out = gr.Image(label="表情", height=400)
        def on_load():
            """每次打开/刷新页面时清空历史（仅清空对话区，避免影响表情渲染）"""
            global history
            history.clear()
            return ""

        inp.submit(chat_wrapper, inputs=[inp], outputs=[chat_out, face_out])
        btn.click(chat_wrapper, inputs=[inp], outputs=[chat_out, face_out])
        demo.load(on_load, outputs=[chat_out])

    print(f"请打开 http://127.0.0.1:{port}", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
