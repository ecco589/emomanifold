#!/usr/bin/env python3
"""M1: 流式 ERNIE + VAD，Gradio UI，悬停 token 看 VAD"""
import sys
import warnings
warnings.filterwarnings("ignore", module="urllib3")
from pathlib import Path
import html

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.stream_ernie_vad import load_chat_model_and_projector, generate_with_vad_stream

history = []


TOOLTIP_CSS = """
<style>.vad{position:relative;cursor:help;border-bottom:1px dotted #999}
.vad:hover::after{content:attr(data-vad);position:absolute;left:0;bottom:100%;background:#333;color:#fff;padding:4px 8px;font-size:12px;white-space:nowrap;z-index:9999;border-radius:4px;margin-bottom:4px}</style>
"""

def to_html(msg, tokens):
    parts = [f"<b>你:</b> {html.escape(msg)}<br><b>ERNIE:</b> "]
    for t, v in tokens:
        vv, aa = v.get("valence", 0), v.get("arousal", 0)
        parts.append(f'<span class="vad" data-vad="V:{vv:.2f} A:{aa:.2f}">{html.escape(t)}</span>')
    return TOOLTIP_CSS + "".join(parts)


def chat(msg):
    if not msg.strip():
        return ""
    tokens = []
    for t, v in generate_with_vad_stream(model, tokenizer, projector, [{"role": "user", "content": msg}], device=device):
        tokens.append((t, v))
        yield "\n\n".join(to_html(m, ts) for m, ts in history + [(msg, tokens)])
    history.append((msg, tokens))


if __name__ == "__main__":
    import gradio as gr

    print("加载模型...")
    model, tokenizer, projector = load_chat_model_and_projector("ernie")
    device = str(next(model.parameters()).device)
    print("请打开 http://127.0.0.1:7860")

    gr.Interface(
        fn=chat, inputs="textbox", outputs="html", title="ERNIE + VAD",
        allow_flagging="never"
    ).launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
