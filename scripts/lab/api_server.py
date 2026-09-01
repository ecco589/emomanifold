#!/usr/bin/env python3
"""Lab inference API — streams tokens + VAD; browser only renders Grimace."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stream_engine import generate_with_vad_stream, load_chat_model_and_projector  # noqa: E402

app = FastAPI(title="EmoManifold API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = _tokenizer = _projector = None


@app.on_event("startup")
def _load():
    global _model, _tokenizer, _projector
    device = os.environ.get("EMO_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    _model, _tokenizer, _projector = load_chat_model_and_projector(device=device)
    print(f"Model ready on {device}", flush=True)


@app.get("/health")
def health():
    return {"ok": _model is not None}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = Field(64, ge=1, le=256)
    temperature: float = Field(0.4, ge=0, le=2)


async def _sse_stream(req: ChatRequest) -> AsyncIterator[str]:
    messages = [m.model_dump() for m in req.messages]
    for token, vad in generate_with_vad_stream(
        _model,
        _tokenizer,
        _projector,
        messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        device=str(next(_model.parameters()).device),
    ):
        payload = {
            "text": token,
            "valence": vad.get("valence"),
            "arousal": vad.get("arousal"),
            "dominance": vad.get("dominance"),
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(_sse_stream(req), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("EMO_API_PORT", "8788"))
    uvicorn.run(app, host="0.0.0.0", port=port)
