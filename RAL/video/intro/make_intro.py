#!/usr/bin/env python3
"""Render the introduction clip from the paper figures and one robot recording.

Outputs (next to this file):
  intro_vo.mp4      burned-in subtitles + voice-over
  intro_silent.mp4  burned-in subtitles, no audio
  intro.srt         subtitle track

Voice-over uses OpenAI gpt-4o-mini-tts (key read from RAL/openai_key) and is
cached in vo/<voice>/ so re-renders do not call the API again. `--tts say`
falls back to the macOS voice.

Usage:  python3 RAL/video/intro/make_intro.py [--preview] [--voice marin] [--tts openai|say]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RAL = HERE.parents[1]
FIG = RAL / "versions/2026-11-05/fig"
CLIP = RAL / "experiment/C/C1/result/multicond_noanchor_20260731/ours/012"
GRIMACE = HERE / "grimace/videos/ours/012.mp4"
DEMO_URL = "emo-h2a.pages.dev"
W, H, FPS = 1920, 1080, 30
FADE = 0.4
PAD = 0.45

BLUE = (31, 111, 178)
RED = (196, 72, 52)
INK = (34, 38, 46)
GREY = (120, 126, 136)
LIGHT = (205, 210, 218)
BG = (255, 255, 255)

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
REG = lambda s: ImageFont.truetype(FONT, s, index=0)
BOLD = lambda s: ImageFont.truetype(FONT, s, index=1)

TITLE = "Emo-H2A"
SUBTITLE = "Driving a Facial Robot Directly from LLM Hidden States at Token Rate"

TTS_INSTRUCTIONS = (
    "Warm, confident narrator for a short robotics research film. "
    "Natural conversational pace, clear American English, light emphasis on numbers, no exaggeration."
)

LINES = {
    "hook": "When a robot talks, its face should move with what it says.",
    "demo": "Try it yourself: the live demo runs in your browser.",
    "old": "LLM-driven robot faces wait for the whole reply, "
    "then show one emotion label for the entire turn.",
    "new": "Emo-H2A removes the label, and drives the face from the model's "
    "hidden states, token by token.",
    "axes": "Affective axes are built once from the model's own vocabulary, "
    "with no training.",
    "proj": "At every token, the hidden state is projected onto valence, arousal, and dominance.",
    "fkcn": "A fuzzy mapping turns this continuous affect into smoothly blended expressions "
    "across 29 actuators.",
    "result": "The face follows the turns inside a reply, "
    "and first moves in under one second.",
}
SPOKEN = {"fkcn": LINES["fkcn"].replace("29", "twenty-nine")}

FIG_TOP, FIG_BOTTOM = 70, 860
AREA = (40, FIG_TOP, 1880, FIG_BOTTOM)

SPEED = 1.4
FACE_SIZE = 780


# --------------------------------------------------------------------------- utils
def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fit(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    s = min(box_w / im.width, box_h / im.height)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def rgba(col, a: float):
    return col + (round(255 * max(0.0, min(1.0, a))),)


def draw_caption(canvas: Image.Image, text: str, alpha: float) -> None:
    if not text or alpha <= 0:
        return
    font = REG(46)
    lines = wrap(text, font, 1740)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = 905 if len(lines) > 1 else 935
    for ln in lines:
        d.text(((W - font.getlength(ln)) / 2, y), ln, font=font, fill=rgba(INK, alpha))
        y += 62
    canvas.alpha_composite(layer)


def place(canvas: Image.Image, im: Image.Image, cx: float, top: float) -> tuple[int, int]:
    x, y = round(cx - im.width / 2), round(top)
    canvas.paste(im, (x, y))
    return x, y


def spotlight(canvas: Image.Image, origin, scale, boxes, amount: float) -> None:
    """Fade everything but `boxes` (figure px) toward white and outline them."""
    if amount <= 0:
        return
    ox, oy = origin
    veil = Image.new("L", canvas.size, round(115 * amount))
    d = ImageDraw.Draw(veil)
    rects = []
    for x0, y0, x1, y1 in boxes:
        r = (ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale)
        rects.append(r)
        d.rounded_rectangle(r, radius=18, fill=0)
    white = Image.new("RGBA", canvas.size, BG + (255,))
    white.putalpha(veil)
    canvas.alpha_composite(white)
    o = ImageDraw.Draw(canvas)
    for r in rects:
        o.rounded_rectangle(r, radius=18, outline=rgba(BLUE, amount), width=5)


def load_flat(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", im.size, BG + (255,))
    return Image.alpha_composite(flat, im).convert("RGB")


# --------------------------------------------------------------------------- assets
fig1 = load_flat(FIG / "大图一.png")
fig2 = load_flat(FIG / "大图二.png")
PANEL_B_SRC = fig1.crop((2180, 100, 4647, 1520))
PANEL_B = fit(PANEL_B_SRC, 1250, FIG_BOTTOM - FIG_TOP)
PANEL_B_SCALE = PANEL_B.width / PANEL_B_SRC.width

B_TOP = (0, 40, 2467, 700)
B_BOTTOM = (0, 740, 2467, 1410)
F2_OFFLINE = (3350, 10, 4545, 1290)
F2_HIDDEN = (1380, 815, 2030, 1085)
F2_VAD = (2520, 40, 3230, 650)
F2_FKCN = (1570, 40, 2390, 650)
F2_FACE = (530, 40, 1500, 650)

def load_clip():
    """Grimace replay of the trial (starts at the first token) and its token stream."""
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(GRIMACE),
         "-vf", f"scale={FACE_SIZE}:{FACE_SIZE},fps={FPS}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    frames = np.frombuffer(raw, np.uint8).reshape(-1, FACE_SIZE, FACE_SIZE, 3)
    tokens = []
    with open(CLIP / "source_va.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tokens.append((float(r["timestamp_s"]), r["token"], float(r["valence"])))
    return frames, tokens


CLIP_FRAMES, CLIP_TOKENS = load_clip()
CLIP_T0 = CLIP_TOKENS[0][0]
TOKEN_SPAN = CLIP_TOKENS[-1][0] - CLIP_T0
VIDEO_SRC = len(CLIP_FRAMES) / FPS
CLIP_LEN = VIDEO_SRC / SPEED


# --------------------------------------------------------------------------- scenes
def scene_cold(t: float, dur: float) -> Image.Image:
    """Grimace face driven token by token, with the reply and valence readout beside it."""
    c = Image.new("RGBA", (W, H), BG + (255,))
    k = min(int(t * SPEED * FPS), len(CLIP_FRAMES) - 1)
    face = Image.fromarray(CLIP_FRAMES[k])
    d = ImageDraw.Draw(c)
    d.rounded_rectangle((90, 70, 90 + FACE_SIZE, 70 + FACE_SIZE), radius=28, outline=LIGHT, width=3)
    mask = Image.new("L", face.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((3, 3, face.width - 3, face.height - 3), radius=26, fill=255)
    c.paste(face, (90, 70), mask)

    a = ease((t - 1.2) / 0.6)
    badge = f"Live demo  ·  {DEMO_URL}"
    bf = BOLD(34)
    bw = bf.getlength(badge) + 56
    bx, by = 970, 818
    d.rounded_rectangle((bx, by, bx + bw, by + 64), radius=32, fill=rgba(BLUE, a))
    d.text((bx + 28, by + 13), badge, font=bf, fill=rgba((255, 255, 255), a))

    clock = CLIP_T0 + t * SPEED
    shown = [tok for tok in CLIP_TOKENS if tok[0] <= clock]
    x0, x1 = 970, 1840
    d.text((x0, 90), "LLM reply, token by token", font=REG(30), fill=GREY)

    font = REG(46)
    x, y = x0, 150
    for i, (ts, tok, v) in enumerate(shown):
        piece = tok.replace("\n", " ")
        lead = piece.startswith(" ")
        word = piece.strip()
        if not word:
            continue
        gap = font.getlength(" ") if lead and x > x0 else 0
        wlen = font.getlength(word)
        if x + gap + wlen > x1:
            x, y = x0, y + 62
            gap = 0
        latest = i == len(shown) - 1
        col = BLUE if latest else INK
        d.text((x + gap, y), word, font=BOLD(46) if latest else font, fill=col)
        x += gap + (BOLD(46).getlength(word) if latest else wlen)

    gx0, gx1, gy0, gy1 = x0, x1, 560, 800
    d.text((gx0, gy0 - 50), "Valence read from the hidden state", font=REG(30), fill=GREY)
    zero = (gy0 + gy1) / 2
    d.line((gx0, zero, gx1, zero), fill=LIGHT, width=2)
    d.text((gx1 + 10, gy0 - 14), "+", font=REG(30), fill=GREY)
    d.text((gx1 + 12, gy1 - 22), "–", font=REG(30), fill=GREY)
    span = TOKEN_SPAN
    pts = [
        (gx0 + (ts - CLIP_T0) / span * (gx1 - gx0), zero - v * (gy1 - gy0) / 2)
        for ts, _, v in CLIP_TOKENS
        if CLIP_T0 <= ts <= clock
    ]
    if len(pts) > 1:
        d.line(pts, fill=BLUE, width=5, joint="curve")
    if pts:
        px, py = pts[-1]
        d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=BLUE)
    return c


def scene_title(t: float, dur: float) -> Image.Image:
    c = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(c)
    a = ease(t / 0.6)
    for text, font, y, col in ((TITLE, BOLD(130), 300, BLUE), (SUBTITLE, REG(50), 485, INK)):
        d.text(((W - font.getlength(text)) / 2, y + 20 * (1 - a)), text, font=font, fill=rgba(col, a))
    b = ease((t - 0.5) / 0.6)
    label, url = "Live demo", DEMO_URL
    lf, uf = REG(36), BOLD(52)
    d.text(((W - lf.getlength(label)) / 2, 620), label, font=lf, fill=rgba(GREY, b))
    d.text(((W - uf.getlength(url)) / 2, 670), url, font=uf, fill=rgba(BLUE, b))
    return c


def _panel_b(t: float, boxes, side_label, value: str, value_col):
    c = Image.new("RGBA", (W, H), BG + (255,))
    origin = place(c, PANEL_B, 820, FIG_TOP)
    spotlight(c, origin, PANEL_B_SCALE, boxes, ease(t / 0.5))
    d = ImageDraw.Draw(c)
    a = ease((t - 0.3) / 0.5)
    d.text((1500, 300), side_label, font=REG(38), fill=rgba(GREY, a))
    d.text((1500, 350), value, font=BOLD(96), fill=rgba(value_col, a))
    return c


def scene_old(t: float, dur: float) -> Image.Image:
    p = ease((t - 0.3) / (dur * 0.6))
    value = "10–26 s" if p >= 1 else f"{10 * p:4.1f} s"
    return _panel_b(t, [B_TOP], "Face waits", value, RED)


def scene_new(t: float, dur: float) -> Image.Image:
    return _panel_b(t, [B_BOTTOM], "First update", "0.96 s", BLUE)


FIG2 = fit(fig2, AREA[2] - AREA[0], AREA[3] - AREA[1])
FIG2_SCALE = FIG2.width / fig2.width


def _fig2(t: float, boxes) -> Image.Image:
    c = Image.new("RGBA", (W, H), BG + (255,))
    origin = place(c, FIG2, W / 2, FIG_TOP + (FIG_BOTTOM - FIG_TOP - FIG2.height) / 2)
    spotlight(c, origin, FIG2_SCALE, boxes, ease(t / 0.5))
    return c


def scene_axes(t, dur):
    return _fig2(t, [F2_OFFLINE])


def scene_proj(t, dur):
    return _fig2(t, [F2_HIDDEN, F2_VAD])


def scene_fkcn(t, dur):
    return _fig2(t, [F2_FKCN, F2_FACE])


def scene_result(t: float, dur: float) -> Image.Image:
    c = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(c)
    cards = (
        ("87.5%", "of within-reply valence transitions", "tracked on the robot face", ease(t / 0.6)),
        ("0.96 s", "to the first facial update,", "vs 10–26 s for turn-level labels", ease((t - 0.25) / 0.6)),
    )
    for i, (big, l1, l2, a) in enumerate(cards):
        cx = W / 2 + (i - 0.5) * 760
        y = 260 + 25 * (1 - a)
        for text, font, dy, col in ((big, BOLD(150), 0, BLUE), (l1, REG(38), 210, INK), (l2, REG(38), 262, GREY)):
            d.text((cx - font.getlength(text) / 2, y + dy), text, font=font, fill=rgba(col, a))
    return c


SCENES = [
    ("cold", scene_cold, "hook"),
    ("title", scene_title, "demo"),
    ("old", scene_old, "old"),
    ("new", scene_new, "new"),
    ("axes", scene_axes, "axes"),
    ("proj", scene_proj, "proj"),
    ("fkcn", scene_fkcn, "fkcn"),
    ("result", scene_result, "result"),
]
FIXED = {"cold": CLIP_LEN, "title": 3.2}
VO_OFFSET = {"cold": 1.6, "title": 0.6}
NO_CROSSFADE: set[str] = set()


# --------------------------------------------------------------------------- audio
def synth(tts: str, voice: str, tmp: Path) -> dict[str, tuple[Path, float]]:
    cache = HERE / "vo" / (voice if tts == "openai" else "say")
    cache.mkdir(parents=True, exist_ok=True)
    client = None
    out = {}
    for key, text in LINES.items():
        spoken = SPOKEN.get(key, text)
        tag = hashlib.sha1(f"{tts}|{voice}|{TTS_INSTRUCTIONS}|{spoken}".encode()).hexdigest()[:10]
        wav = cache / f"{key}_{tag}.wav"
        if not wav.exists():
            if tts == "openai":
                if client is None:
                    from openai import OpenAI

                    key_file = RAL / "openai_key"
                    client = OpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY") or key_file.read_text().strip(),
                        max_retries=6,
                        timeout=90,
                    )
                with client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts", voice=voice, input=spoken,
                    instructions=TTS_INSTRUCTIONS, response_format="wav",
                ) as r:
                    r.stream_to_file(wav)
            else:
                subprocess.run(["say", "-v", "Samantha", "-r", "172", "-o", str(wav),
                                "--data-format=LEI16@44100", spoken], check=True)
        trimmed = wav.with_suffix(".trim.wav")
        if not trimmed.exists():
            edge = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-af",
                 f"{edge},areverse,{edge},areverse", str(trimmed)],
                check=True,
            )
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(trimmed)]
        ))
        out[key] = (trimmed, dur)
    return out


def srt_time(s: float) -> str:
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="dump one still per scene")
    ap.add_argument("--tts", choices=("openai", "say"), default="openai")
    ap.add_argument("--voice", default="marin")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        audio = synth(args.tts, args.voice, Path(td))

        timeline, t0 = [], 0.0
        for name, fn, key in SCENES:
            if name in FIXED:
                dur = FIXED[name]
                if key:
                    dur = max(dur, VO_OFFSET.get(name, 0.25) + audio[key][1] + 0.3)
            else:
                dur = audio[key][1] + PAD + (0.8 if name == "result" else 0)
            timeline.append((name, fn, key, t0, dur))
            t0 += dur
        total = t0

        if args.preview:
            pv = HERE / "preview"
            pv.mkdir(exist_ok=True)
            for name, fn, key, start, dur in timeline:
                fr = fn(dur * 0.85, dur)
                draw_caption(fr, LINES.get(key, ""), 1.0)
                fr.convert("RGB").save(pv / f"{name}.png")
            print(f"preview written ({total:.1f} s)")
            return

        silent = HERE / "intro_silent.mp4"
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)],
            stdin=subprocess.PIPE,
        )
        n = round(total * FPS)
        last_frames: dict[int, Image.Image] = {}
        for i in range(n):
            t = i / FPS
            idx = max(j for j, s in enumerate(timeline) if s[3] <= t + 1e-9)
            name, fn, key, start, dur = timeline[idx]
            local = t - start
            frame = fn(local, dur)
            cap_a = min(ease(local / FADE), ease((dur - local) / FADE))
            if key in VO_OFFSET or name in VO_OFFSET:
                cap_a = min(cap_a, ease((local - VO_OFFSET.get(name, 0) + 0.2) / FADE))
            draw_caption(frame, LINES.get(key, ""), cap_a)
            if local < FADE and idx > 0 and name not in NO_CROSSFADE:
                prev = last_frames[idx - 1]
                frame = Image.blend(prev.convert("RGB"), frame.convert("RGB"), ease(local / FADE))
            frame = frame.convert("RGB")
            last_frames[idx] = frame
            ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        ff.stdin.close()
        ff.wait()

        inputs, filters = [], []
        voiced = [s for s in timeline if s[2]]
        for k, (name, fn, key, start, dur) in enumerate(voiced):
            inputs += ["-i", str(audio[key][0])]
            ms = round((start + VO_OFFSET.get(name, 0.25)) * 1000)
            filters.append(f"[{k + 1}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
        mix = "".join(f"[a{k}]" for k in range(len(voiced)))
        filters.append(f"{mix}amix=inputs={len(voiced)}:normalize=0,apad[aout]")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs,
             "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             str(HERE / "intro_vo.mp4")],
            check=True,
        )

        with open(HERE / "intro.srt", "w", encoding="utf-8") as f:
            for k, (name, fn, key, start, dur) in enumerate(voiced, 1):
                s0 = start + VO_OFFSET.get(name, 0.25)
                f.write(f"{k}\n{srt_time(s0)} --> {srt_time(s0 + audio[key][1] + 0.2)}\n{LINES[key]}\n\n")
        print(f"done: {total:.1f} s, {n} frames")


if __name__ == "__main__":
    main()
