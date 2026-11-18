#!/usr/bin/env python3
"""Xpress storytelling programs for the 20 C1 replies.

Follows vantony1/Xpress end2end up to the timestamped emotion program:
  TTS -> whisper-1 word timestamps -> processTranscript(group=1)
  -> segmentStory (temperature 0.1)
  -> story2emotion (temperature 0.3)

gpt-4-turbo, the model named in the Xpress code, is retired on this
account. Both calls use gpt-4o, the current replacement, at the original
temperatures.

Their face2code targets a 28-DoF HTML face. This script stops at the
emotion timeline, which is the program the physical face can play.
TTS is OpenAI tts-1; the original demo used Google TTS, which this key
cannot call. The language models are the original ones.
"""
from __future__ import annotations

import ast
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
PROMPTS = Path(__file__).resolve().parent / "xpress_story_prompts"
KEY_FILE = ROOT / "RAL" / "openai_key"
FIELD = ROOT / "RAL/experiment/C/C1/result/multicond_noanchor_20260731/ours"
OUT = ROOT / "RAL/experiment/C/C1/result/xpress_story_20261111"
LABEL_MAP = json.loads(
    (Path(__file__).resolve().parent / "xpress_label_map.json").read_text(encoding="utf-8")
)

LABEL_VA = {
    "happiness": (0.85, 0.62),
    "sadness": (-0.50, -0.57),
    "fear": (-0.87, 0.53),
    "anger": (-0.96, 0.13),
    "surprise": (-0.73, 0.60),
    "disgust": (-0.90, -0.12),
    "neutral": (0.61, -0.47),
}


def key() -> str:
    return KEY_FILE.read_text(encoding="utf-8").strip()


def api(path: str, payload: dict | None = None, files: dict | None = None) -> bytes:
    token = key()
    url = "https://api.openai.com/v1" + path
    for attempt in range(5):
        try:
            if files is None:
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
            else:
                boundary = "----xpressstoryboundary"
                body = _multipart(boundary, files)
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise RuntimeError(f"HTTP {e.code} {path}: {err}") from e
        except urllib.error.URLError:
            if attempt < 4:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    raise RuntimeError(path)


def _multipart(boundary: str, fields: dict) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):
            filename, content, mime = value
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
            chunks.append(content)
            chunks.append(b"\r\n")
        else:
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            chunks.append(str(value).encode())
            chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def chat(messages: list, temperature: float) -> str:
    raw = api(
        "/chat/completions",
        {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,
        },
    )
    body = json.loads(raw)
    return body["choices"][0]["message"]["content"]


def parse_obj(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1]
        s = s.rsplit("```", 1)[0].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = ast.literal_eval(s)
    if not isinstance(obj, dict):
        raise ValueError(f"expected object, got {type(obj).__name__}")
    return obj


def process_transcript(words: list, group_size: int = 1) -> str:
    parts = []
    for i in range(0, len(words) - group_size + 1, group_size):
        group = words[i : i + group_size]
        text = " ".join(w["word"].strip() for w in group)
        ts = float(group[-1]["start"])
        ts = int(ts * 100) / 100
        parts.append(f"{text} <{ts:.2f}>")
    return " ".join(parts)


def one_trial(tid: str, assistant_text: str) -> None:
    dest = OUT / tid
    dest.mkdir(parents=True, exist_ok=True)
    audio = api(
        "/audio/speech",
        {"model": "tts-1", "voice": "alloy", "input": assistant_text},
    )
    (dest / "speech.mp3").write_bytes(audio)
    transcript = json.loads(
        api(
            "/audio/transcriptions",
            files={
                "file": ("speech.mp3", audio, "audio/mpeg"),
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )
    )
    words = transcript.get("words") or []
    if not words:
        raise RuntimeError(f"{tid}: whisper returned no words")
    tagged = process_transcript(words, 1)
    (dest / "tagged.txt").write_text(tagged + "\n", encoding="utf-8")
    (dest / "whisper.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    segment_prompt = json.loads((PROMPTS / "segmentStory.json").read_text(encoding="utf-8"))
    emotion_prompt = json.loads((PROMPTS / "story2Emotion.json").read_text(encoding="utf-8"))
    segmented = parse_obj(chat([*segment_prompt, {"role": "user", "content": tagged}], 0.1))
    emotions = parse_obj(chat([*emotion_prompt, {"role": "user", "content": tagged}], 0.3))
    (dest / "segment.json").write_text(
        json.dumps(segmented, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "emotion.json").write_text(
        json.dumps(emotions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    scenes = []
    for name, scene in emotions.items():
        raw = str(scene["emotion"]).strip().lower()
        mapped = LABEL_MAP.get(raw, "neutral")
        v, a = LABEL_VA[mapped]
        scenes.append(
            {
                "scene": name,
                "timestamp_s": float(scene["start_time"]),
                "emotion_raw": raw,
                "emotion_name": mapped,
                "valence": v,
                "arousal": a,
            }
        )
    scenes.sort(key=lambda r: r["timestamp_s"])
    audio_end = float(words[-1]["end"])
    lines = ["timestamp_s,valence,arousal,token"]
    for row in scenes:
        lines.append(
            f"{row['timestamp_s']:.4f},{row['valence']},{row['arousal']},"
            f"LABEL:{row['emotion_name']}:{row['emotion_raw']}"
        )
    (dest / "source_va.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "trial_id": tid,
        "assistant_text": assistant_text,
        "tts_model": "tts-1",
        "tts_note": "OpenAI tts-1 stands in for the original Google TTS",
        "whisper_model": "whisper-1",
        "segment_model": "gpt-4o",
        "segment_model_note": "gpt-4-turbo is retired; gpt-4o at the original temperature",
        "segment_temperature": 0.1,
        "emotion_model": "gpt-4o",
        "emotion_temperature": 0.3,
        "audio_duration_s": audio_end,
        "n_keyframes": len(scenes),
        "keyframes": scenes,
    }
    (dest / "program.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{tid}  {audio_end:.1f}s  {len(scenes)} keyframes  "
        + " ".join(f"{r['timestamp_s']:.1f}:{r['emotion_raw']}" for r in scenes),
        flush=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in range(1, 21):
        tid = f"{n:03d}"
        if (OUT / tid / "source_va.csv").is_file():
            print(f"{tid} exists, skip", flush=True)
            continue
        session = json.loads((FIELD / tid / "session.json").read_text(encoding="utf-8"))
        text = session["assistant_text"].strip()
        one_trial(tid, text)


if __name__ == "__main__":
    main()
