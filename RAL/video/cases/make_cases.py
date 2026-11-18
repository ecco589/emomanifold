#!/usr/bin/env python3
"""Render the case studies and ending that follow the intro, then join them with it.

Sequence:
  c1a     EmoManifold alone: chat bubbles, graded robot face, valence/arousal curves
  bridge  one-line card into the comparison
  c1b     the same reply under five conditions, each with its valence command
  c2      time to the first facial update (stopwatch, x4 during the wait)
  outro   four more replies + three takeaways, then the title card

Outputs (next to this file):
  cases_vo.mp4, cases_silent.mp4, cases.srt
  ../promo.mp4, ../promo.srt   intro + cases

Usage:  python3 RAL/video/cases/make_cases.py [--preview] [--voice marin] [--no-join]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
RAL = HERE.parents[1]
REC = RAL / "newmainexp"
OURS = REC / "emomanifold" / "004*"
CONDS = {
    "ours": OURS,
    "story": REC / "xpress_story" / "004",
    "turn": REC / "xpress_turn" / "004",
    "mishra": REC / "mishra" / "004",
    "shuffled": REC / "shuffled" / "004",
}
MONTAGE = [
    ("001", "I just won the lottery and I'm so excited!"),
    ("005", "That person bullied me and I'm furious!"),
    ("008", "That smell is disgusting and I can't stand it."),
    ("006", "The thunder outside is so loud and I'm a bit scared."),
]
DEMO_URL = "emo-h2a.pages.dev"
W, H, FPS = 1920, 1080, 30
FADE = 0.4
AUDIO_DELAY = 0.5  # speech is embedded 0.5 s after the token clock

# Mean first-update latency on the onboard PC, local model (paper latency table).
LATENCY = {"ours": 0.96, "mishra": 9.9, "turn": 26.5, "story": 93.5}

CROP = "crop=1000:1000:740:110"
GRADE = ("hqdn3d=1.2:1.2:2:2,curves=master='0/0 0.2/0.18 0.8/0.84 1/1',eq=saturation=1.06,"
         "colorbalance=rm=0.012:bm=-0.015:bh=0.015,unsharp=5:5:0.45,vignette=angle=PI/7")

BLUE = (31, 111, 178)
ORANGE = (222, 132, 44)
RED = (196, 72, 52)
INK = (34, 38, 46)
GREY = (120, 126, 136)
LIGHT = (205, 210, 218)
PALE = (238, 240, 244)
BUBBLE = (230, 240, 250)
BG = (255, 255, 255)

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
_fonts: dict = {}


def REG(s):
    return _fonts.setdefault(("r", s), ImageFont.truetype(FONT, s, index=0))


def BOLD(s):
    return _fonts.setdefault(("b", s), ImageFont.truetype(FONT, s, index=1))


TITLE = "Emo-H2A"
SUBTITLE = "Driving a Facial Robot Directly from LLM Hidden States at Token Rate"

TTS_INSTRUCTIONS = (
    "Warm, confident narrator for a short robotics research film. "
    "Natural conversational pace, clear American English, light emphasis on numbers, no exaggeration."
)
LINES = {
    "bridge": "Here is the same reply, driven five different ways.",
    "c2a": "How soon does the face react? EmoManifold moves in under one second.",
    "c2b": "Turn-level systems wait for the whole reply, then for a label.",
    "c2c": "Xpress-Story plans the whole reply first: over ninety seconds.",
    "end": "Emo-H2A needs no training, follows the reply token by token, "
    "and observers preferred it in 83.5% of trials.",
    "outro": "Let the face move with what the robot says.",
}
SPOKEN = {"end": LINES["end"].replace("83.5%", "eighty-three point five percent")}


# --------------------------------------------------------------------------- utils
def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def rgba(col, a: float):
    return col + (round(255 * max(0.0, min(1.0, a))),)


def wrap(text: str, font, width: int) -> list[str]:
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


def paste_round(canvas: Image.Image, im: Image.Image, xy, radius=24, border=None, width=0) -> None:
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=radius, fill=255)
    canvas.paste(im, xy, mask)
    if border:
        x, y = xy
        ImageDraw.Draw(canvas).rounded_rectangle(
            (x - width // 2, y - width // 2, x + im.width + width // 2, y + im.height + width // 2),
            radius=radius + width // 2, outline=border, width=width)


def header(d: ImageDraw.ImageDraw, tag: str, text: str, alpha: float = 1.0, sub: str = "") -> None:
    tf = BOLD(28)
    tw = tf.getlength(tag) + 36
    d.rounded_rectangle((80, 42, 80 + tw, 88), radius=23, fill=rgba(BLUE, alpha))
    d.text((98, 50), tag, font=tf, fill=rgba((255, 255, 255), alpha))
    d.text((80 + tw + 22, 42), text, font=BOLD(40), fill=rgba(INK, alpha))
    if sub:
        d.text((80 + tw + 22, 92), sub, font=REG(28), fill=rgba(GREY, alpha))


def blank() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    c = Image.new("RGBA", (W, H), BG + (255,))
    return c, ImageDraw.Draw(c)


class Clip:
    """Sequential frame reader (cropped, graded, square); holds the last frame at the end."""

    def __init__(self, path: Path, size: int, start: float = 0.0):
        self.size = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path),
             "-vf", f"{CROP},{GRADE},scale={size}:{size},fps={FPS}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        self.idx, self.frame, self.done = -1, None, False
        self.at(0.0)

    def at(self, t: float) -> Image.Image:
        k = max(0, int(t * FPS + 1e-6))
        nbytes = self.size * self.size * 3
        while self.idx < k and not self.done:
            buf = self.proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                self.done = True
                self.proc.stdout.close()
                break
            self.frame = Image.frombuffer("RGB", (self.size, self.size), buf)
            self.idx += 1
        return self.frame


def read_tokens(path: Path) -> list[tuple[float, str, float, float]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((float(r["t"]) + AUDIO_DELAY, r["token"], float(r["v"]), float(r["a"] or 0)))
    return out


TOKENS = read_tokens(OURS / "playback_va.csv")
SESSION = json.loads((OURS / "session.json").read_text())
USER_PROMPT = SESSION["user_prompt"]
COMMANDS = {k: read_tokens(p / "playback_va.csv") for k, p in CONDS.items()}
OURS_LEN = float(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
     str(OURS / "video.mp4")]))


# --------------------------------------------------------------------------- c1a
C1A_LEAD = 1.8
C1A_DUR = C1A_LEAD + OURS_LEN + 1.2
FACE1 = 540
PLOT_X0, PLOT_X1 = 720, 1840
PLOT_V = (540, 770)
PLOT_A = (815, 1045)
PLOT_T1 = TOKENS[-1][0] + 0.4

BUBBLE_X1 = 1840
BUBBLE_W = 1300
BUBBLE_TOP = 222
TEXT_FONT = REG(34)
TEXT_BOLD = BOLD(34)
LINE_H = 46


def layout_reply() -> tuple[list[tuple[int, float, float, str]], int]:
    """Fixed positions for every token so the reply never reflows while it streams."""
    x0 = BUBBLE_X1 - BUBBLE_W + 30
    x1 = BUBBLE_X1 - 30
    x, y, pos = x0, BUBBLE_TOP + 24, []
    for i, (_, tok, _, _) in enumerate(TOKENS):
        word = tok.strip()
        if not word:
            continue
        gap = TEXT_FONT.getlength(" ") if tok.startswith(" ") and x > x0 else 0
        wlen = TEXT_BOLD.getlength(word)
        if x + gap + wlen > x1:
            x, y, gap = x0, y + LINE_H, 0
        pos.append((i, x + gap, y, word))
        x += gap + TEXT_FONT.getlength(word)
    return pos, y + LINE_H + 22


REPLY_POS, BUBBLE_BOTTOM = layout_reply()
SYMPATHY = min(range(12), key=lambda i: TOKENS[i][2])
REASSURE = max(range(10, 20), key=lambda i: TOKENS[i][2])


def plot_frame(d, top_bot, label, col):
    y0, y1 = top_bot
    zero = (y0 + y1) / 2
    d.text((PLOT_X0, y0 - 6), label, font=BOLD(28), fill=col)
    d.line((PLOT_X0, zero, PLOT_X1, zero), fill=LIGHT, width=2)
    d.text((PLOT_X1 + 10, y0 + 22), "+", font=REG(30), fill=GREY)
    d.text((PLOT_X1 + 12, y1 - 40), "–", font=REG(30), fill=GREY)


def plot_xy(t, val, top_bot):
    y0, y1 = top_bot
    y0 += 34
    zero = (y0 + y1) / 2
    x = PLOT_X0 + t / PLOT_T1 * (PLOT_X1 - PLOT_X0)
    return x, zero - val * (y1 - y0) / 2


def note(d, xy, text, alpha):
    if alpha <= 0:
        return
    x, y = xy
    d.ellipse((x - 13, y - 13, x + 13, y + 13), outline=rgba(INK, alpha), width=3)
    d.text((x + 22, y - 15), text, font=BOLD(26), fill=rgba(INK, alpha))


def scene_c1a(t: float, dur: float, st: dict) -> Image.Image:
    if "face" not in st:
        st["face"] = Clip(OURS / "video.mp4", FACE1)
    c, d = blank()
    header(d, "Case 1", "The face follows the reply, token by token", ease(t / 0.5))
    vt = t - C1A_LEAD

    a_user = ease(t / 0.5)
    d.text((80, 112), "User", font=REG(26), fill=rgba(GREY, a_user))
    uw = TEXT_FONT.getlength(USER_PROMPT) + 48
    d.rounded_rectangle((80, 148, 80 + uw, 208), radius=26, fill=rgba(PALE, a_user))
    d.text((104, 158), USER_PROMPT, font=TEXT_FONT, fill=rgba(INK, a_user))

    a_bot = ease((t - C1A_LEAD + 0.6) / 0.5)
    rx0 = BUBBLE_X1 - BUBBLE_W
    d.text((BUBBLE_X1 - REG(26).getlength("Robot"), BUBBLE_TOP - 36), "Robot", font=REG(26),
           fill=rgba(GREY, a_bot))
    d.rounded_rectangle((rx0, BUBBLE_TOP, BUBBLE_X1, BUBBLE_BOTTOM), radius=26, fill=rgba(BUBBLE, a_bot))
    shown = [p for p in REPLY_POS if TOKENS[p[0]][0] <= vt]
    for k, (i, x, y, word) in enumerate(shown):
        latest = k == len(shown) - 1
        d.text((x, y), word, font=TEXT_BOLD if latest else TEXT_FONT, fill=BLUE if latest else INK)

    face = st["face"].at(max(vt, 0.0))
    paste_round(c, face, (80, 1060 - FACE1 - 15), radius=26)

    plot_frame(d, PLOT_V, "Valence", BLUE)
    plot_frame(d, PLOT_A, "Arousal", ORANGE)
    seen = [tok for tok in TOKENS if tok[0] <= vt]
    for idx, box, col in ((2, PLOT_V, BLUE), (3, PLOT_A, ORANGE)):
        pts = [plot_xy(tok[0], tok[idx], box) for tok in seen]
        if len(pts) > 1:
            d.line(pts, fill=col, width=5, joint="curve")
        if pts:
            px, py = pts[-1]
            d.ellipse((px - 8, py - 8, px + 8, py + 8), fill=col)
    if 0 <= vt <= PLOT_T1:
        cx = PLOT_X0 + vt / PLOT_T1 * (PLOT_X1 - PLOT_X0)
        d.line((cx, PLOT_V[0] + 34, cx, PLOT_A[1]), fill=LIGHT, width=2)

    s_t, _, s_v, _ = TOKENS[SYMPATHY]
    r_t, _, r_v, _ = TOKENS[REASSURE]
    note(d, plot_xy(s_t, s_v, PLOT_V), "sympathy", ease((vt - s_t - 0.2) / 0.4))
    note(d, plot_xy(r_t, r_v, PLOT_V), "“It's okay …”  reassurance", ease((vt - r_t - 0.2) / 0.4))
    return c


# --------------------------------------------------------------------------- bridge
def scene_bridge(t: float, dur: float, st: dict) -> Image.Image:
    c, d = blank()
    a = ease(t / 0.5)
    for text, font, y, col in (("Same reply,", BOLD(84), 330, INK),
                               ("five ways to drive the face", BOLD(84), 440, BLUE)):
        d.text(((W - font.getlength(text)) / 2, y + 16 * (1 - a)), text, font=font, fill=rgba(col, a))
    return c


# --------------------------------------------------------------------------- c1b
C1B_LEAD, C1B_END = 0.6, 10.4
C1B_DUR = C1B_LEAD + C1B_END + 0.7
TILE = 330
TILE_GAP = 30
TILE_Y = 250
TILES = [
    ("ours", "EmoManifold", "every token (ours)"),
    ("story", "Xpress-Story", "pre-scripted segments"),
    ("turn", "Xpress-Turn", "one label per reply"),
    ("mishra", "Mishra et al.", "one label per reply"),
    ("shuffled", "Shuffled", "our commands, reordered"),
]
MINI = (TILE_Y + TILE + 40, TILE_Y + TILE + 150)
C1B_T1 = C1B_END


def sentences() -> list[list[int]]:
    out, cur = [], []
    for i, (_, tok, _, _) in enumerate(TOKENS):
        cur.append(i)
        if tok.strip() in (".", "!", "?"):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


SENTENCES = sentences()


def command_at(key: str, t: float) -> float | None:
    rows = [r for r in COMMANDS[key] if r[0] <= t]
    if not rows:
        return COMMANDS[key][0][2] if key in ("turn", "mishra") else None
    return rows[-1][2]


def scene_c1b(t: float, dur: float, st: dict) -> Image.Image:
    if "clips" not in st:
        st["clips"] = {k: Clip(CONDS[k] / "video.mp4", TILE) for k, _, _ in TILES}
    c, d = blank()
    header(d, "Case 1", "Same reply, five ways to drive the face", ease(t / 0.5),
           sub="Below each face: the valence command it receives")
    vt = min(t - C1B_LEAD, C1B_END)
    x0 = (W - (5 * TILE + 4 * TILE_GAP)) // 2
    for n, (key, name, desc) in enumerate(TILES):
        x = x0 + n * (TILE + TILE_GAP)
        ours = key == "ours"
        d.text((x, TILE_Y - 82), name, font=BOLD(32), fill=BLUE if ours else INK)
        d.text((x, TILE_Y - 42), desc, font=REG(25), fill=GREY)
        face = st["clips"][key].at(max(vt, 0.0))
        paste_round(c, face, (x, TILE_Y), radius=20,
                    border=BLUE if ours else LIGHT, width=6 if ours else 2)

        y0, y1 = MINI
        zero = (y0 + y1) / 2
        d.rounded_rectangle((x, y0, x + TILE, y1), radius=14, fill=PALE)
        d.line((x + 10, zero, x + TILE - 10, zero), fill=LIGHT, width=2)
        pts = []
        for k in range(0, 121):
            tt = k / 120 * C1B_T1
            if tt > max(vt, 0):
                break
            v = command_at(key, tt)
            if v is not None:
                pts.append((x + 10 + tt / C1B_T1 * (TILE - 20), zero - v * (y1 - y0 - 16) / 2))
        if len(pts) > 1:
            d.line(pts, fill=BLUE if ours else GREY, width=4)
        if pts:
            px, py = pts[-1]
            d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=BLUE if ours else GREY)

    cur = None
    for k, sent in enumerate(SENTENCES):
        if TOKENS[sent[0]][0] <= max(vt, 0.0) + 0.05:
            cur = k
    if cur is not None and vt >= 0:
        lines, y = SENT_LAYOUT[cur], 780 if len(SENT_LAYOUT[cur]) > 1 else 805
        for ln in lines:
            for i, x, word in ln:
                d.text((x, y), word, font=KARAOKE, fill=INK if TOKENS[i][0] <= vt else LIGHT)
            y += 58
    return c


KARAOKE = REG(42)


def layout_sentence(idx: list[int], width: int = 1600) -> list[list[tuple[int, float, str]]]:
    lines, cur, x = [], [], 0.0
    space = KARAOKE.getlength(" ")
    for i in idx:
        tok = TOKENS[i][1]
        word = tok.strip()
        if not word:
            continue
        gap = space if tok.startswith(" ") and cur else 0
        wlen = KARAOKE.getlength(word)
        if cur and x + gap + wlen > width:
            lines.append(cur)
            cur, x, gap = [], 0.0, 0
        cur.append((i, x + gap, word))
        x += gap + wlen
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        end = ln[-1][1] + KARAOKE.getlength(ln[-1][2])
        off = (W - end) / 2
        out.append([(i, off + x, w) for i, x, w in ln])
    return out


SENT_LAYOUT = [layout_sentence(s) for s in SENTENCES]


# --------------------------------------------------------------------------- c2
# Smooth clock: clock = A (e^(k u) - 1). The four onsets land a few seconds
# apart on screen; the bar below the number shows them on a linear 0–95 s scale.
C2_PRE, C2_SWEEP, C2_STOP, C2_A = 0.55, 12.0, 95.0, 3.0
C2_K = math.log(1 + C2_STOP / C2_A) / C2_SWEEP
C2_END = C2_PRE + C2_SWEEP
C2_DUR = C2_END + 6.0  # leave a beat after the last line before the outro
TILE2 = 390
RACE = [("ours", "EmoManifold"), ("mishra", "Mishra et al."),
        ("turn", "Xpress-Turn"), ("story", "Xpress-Story")]
WAIT = {
    "ours": "",
    "mishra": "waiting for reply + label",
    "turn": "waiting for reply + label",
    "story": "waiting for reply + story plan",
}


def c2_clock(t: float) -> float:
    u = min(max(t - C2_PRE, 0.0), C2_SWEEP)
    return C2_A * math.expm1(C2_K * u)


def c2_onset_t(clock: float) -> float:
    """Scene time at which the clock reads `clock`."""
    return C2_PRE + math.log1p(clock / C2_A) / C2_K


def c2_stopwatch(clock: float, stopped: bool) -> str:
    shown = min(clock, LATENCY["story"])
    if stopped:
        return f"{LATENCY['story']:g} s"
    if shown < 2.0:
        return f"{shown:.1f} s"
    return f"{int(shown)} s"


def draw_c2_bar(d: ImageDraw.ImageDraw, clock: float, a: float) -> None:
    """Linear 0–95 s scale under the stopwatch; ticks at the four onsets."""
    x0, x1, y = 260, 1660, 278
    d.rounded_rectangle((x0, y, x1, y + 10), radius=5, fill=rgba(LIGHT, a))
    fill_x = x0 + (x1 - x0) * min(clock, C2_STOP) / C2_STOP
    if fill_x > x0 + 2:
        d.rounded_rectangle((x0, y, fill_x, y + 10), radius=5, fill=rgba(BLUE, a))
    for lab, xx, align in (("0", x0, "left"), ("95 s", x1, "right")):
        lf = REG(22)
        tw = lf.getlength(lab)
        tx = xx if align == "left" else xx - tw
        d.text((tx, y + 18), lab, font=lf, fill=rgba(GREY, a * 0.85))
    for i, (key, _) in enumerate(RACE):
        onset = LATENCY[key]
        xx = x0 + (x1 - x0) * (onset / C2_STOP)
        hit = clock >= onset
        col = BLUE if key == "ours" else (RED if hit else GREY)
        d.line((xx, y - 10, xx, y + 20), fill=rgba(col, a if hit else a * 0.55), width=3)
        lab = f"{onset:g}"
        lf = BOLD(22) if hit else REG(20)
        tw = lf.getlength(lab)
        ly = y - 36 - (i % 2) * 22
        d.text((xx - tw / 2, ly), lab, font=lf, fill=rgba(col, a if hit else a * 0.5))
    r = 7
    d.ellipse((fill_x - r, y + 5 - r, fill_x + r, y + 5 + r), fill=rgba(BLUE, a))


def scene_c2(t: float, dur: float, st: dict) -> Image.Image:
    if "clips" not in st:
        st["clips"] = {k: Clip(CONDS[k] / "video.mp4", TILE2) for k, _ in RACE}
        st["rest"] = Clip(OURS / "video.mp4", TILE2).at(0).copy()
    c, d = blank()
    ha = ease(t / 0.5)
    header(d, "Case 2", "When does the face start to move?", ha,
           sub="Clock starts when the user finishes speaking")
    clock = c2_clock(t)
    stopped = t >= C2_END - 1e-6
    sw = c2_stopwatch(clock, stopped)
    f = BOLD(96)
    d.text(((W - f.getlength(sw)) / 2, 142), sw, font=f, fill=INK)
    draw_c2_bar(d, clock, ha)

    gap = 36
    x0 = (W - (4 * TILE2 + 3 * gap)) // 2
    ty = 400
    for n, (key, name) in enumerate(RACE):
        x = x0 + n * (TILE2 + gap)
        onset = LATENCY[key]
        moving = clock >= onset
        since = t - c2_onset_t(onset)
        ours = key == "ours"
        d.text((x, ty - 48), name, font=BOLD(32), fill=BLUE if ours else INK)
        if moving:
            face = st["clips"][key].at(max(since, 0.0))
        else:
            face = Image.blend(st["rest"], Image.new("RGB", st["rest"].size, BG), 0.35)
        paste_round(c, face, (x, ty), radius=20,
                    border=BLUE if ours else LIGHT, width=6 if ours else 2)
        col = BLUE if ours else RED
        if moving:
            a = ease(since / 0.45)
            label = f"moves at {onset:g} s"
            lf = BOLD(30)
            d.rounded_rectangle((x, ty + TILE2 + 20, x + lf.getlength(label) + 36, ty + TILE2 + 70),
                                radius=25, fill=rgba(col, a))
            d.text((x + 18, ty + TILE2 + 28), label, font=lf, fill=rgba((255, 255, 255), a))
        else:
            d.text((x, ty + TILE2 + 28), WAIT[key], font=REG(24), fill=GREY)
    note_txt = "Waiting times: mean measured latency on the same onboard PC."
    d.text(((W - REG(22).getlength(note_txt)) / 2, 1042), note_txt, font=REG(22), fill=GREY)
    return c


# --------------------------------------------------------------------------- outro
TILE3 = 380
BULLET_STARTS = [0.6, 2.4, 4.4]
BULLETS = [
    ("Training-free", "axes come from the model's own vocabulary"),
    ("Token-rate", "the face follows the reply as it is generated"),
    ("83.5%", "of observer choices favor the token-level face"),
]


def scene_montage(t: float, dur: float, st: dict) -> Image.Image:
    if "clips" not in st:
        st["clips"] = [Clip(REC / "emomanifold" / tid / "video.mp4", TILE3, start=0.8) for tid, _ in MONTAGE]
    c, d = blank()
    a = ease(t / 0.5)
    gx, gy, gap = 90, 85, 22
    for n, (tid, prompt) in enumerate(MONTAGE):
        x = gx + (n % 2) * (TILE3 + gap)
        y = gy + (n // 2) * (TILE3 + gap)
        face = st["clips"][n].at(t).copy().convert("RGBA")
        shade = Image.new("RGBA", face.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for k in range(90):
            sd.line((0, TILE3 - 90 + k, TILE3, TILE3 - 90 + k), fill=(0, 0, 0, int(150 * k / 90)))
        face.alpha_composite(shade)
        fd = ImageDraw.Draw(face)
        lines = wrap(f"“{prompt}”", REG(22), TILE3 - 28)
        yy = TILE3 - 14 - 28 * len(lines)
        for ln in lines:
            fd.text((14, yy), ln, font=REG(22), fill=(255, 255, 255))
            yy += 28
        paste_round(c, face.convert("RGB"), (x, y), radius=20)

    rx = 1010
    d.text((rx, 150), TITLE, font=BOLD(88), fill=rgba(BLUE, a))
    starts = BULLET_STARTS
    for k, (head, rest) in enumerate(BULLETS):
        b = ease((t - starts[k]) / 0.5)
        y = 330 + k * 150 + 14 * (1 - b)
        d.text((rx, y), head, font=BOLD(54), fill=rgba(INK, b))
        d.text((rx, y + 66), rest, font=REG(32), fill=rgba(GREY, b))
    return c


def scene_title(t: float, dur: float, st: dict) -> Image.Image:
    c, d = blank()
    a = ease(t / 0.6)
    for text, font, y, col in ((TITLE, BOLD(130), 300, BLUE), (SUBTITLE, REG(50), 485, INK)):
        d.text(((W - font.getlength(text)) / 2, y + 20 * (1 - a)), text, font=font, fill=rgba(col, a))
    b = ease((t - 0.5) / 0.6)
    lf, uf = REG(36), BOLD(52)
    d.text(((W - lf.getlength("Live demo")) / 2, 620), "Live demo", font=lf, fill=rgba(GREY, b))
    d.text(((W - uf.getlength(DEMO_URL)) / 2, 670), DEMO_URL, font=uf, fill=rgba(BLUE, b))
    return c


# --------------------------------------------------------------------------- timeline
# name, render fn, base duration, [(vo key, offset)], robot speech (offset, trim end) or None
SCENES = [
    ("c1a", scene_c1a, C1A_DUR, [], (C1A_LEAD, None)),
    ("bridge", scene_bridge, 2.4, [("bridge", 0.35)], None),
    ("c1b", scene_c1b, C1B_DUR, [], (C1B_LEAD, C1B_END)),
    ("c2", scene_c2, C2_DUR, [
        ("c2a", 0.40),
        ("c2b", c2_onset_t(LATENCY["mishra"]) + 0.35),
        ("c2c", c2_onset_t(LATENCY["story"]) - 1.15),
    ], None),
    ("montage", scene_montage, 7.5, [("end", 0.5)], None),
    ("title", scene_title, 3.6, [("outro", 0.6)], None),
]
NO_CROSSFADE = {"c1b"}


# --------------------------------------------------------------------------- audio
def synth(voice: str) -> dict[str, tuple[Path, float]]:
    cache = HERE / "vo" / voice
    cache.mkdir(parents=True, exist_ok=True)
    client = None
    out = {}
    for key, text in LINES.items():
        spoken = SPOKEN.get(key, text)
        tag = hashlib.sha1(f"openai|{voice}|{TTS_INSTRUCTIONS}|{spoken}".encode()).hexdigest()[:10]
        wav = cache / f"{key}_{tag}.wav"
        if not wav.exists():
            if client is None:
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY") or (RAL / "openai_key").read_text().strip(),
                    max_retries=6, timeout=90,
                )
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice=voice, input=spoken,
                instructions=TTS_INSTRUCTIONS, response_format="wav",
            ) as r:
                r.stream_to_file(wav)
        trimmed = wav.with_suffix(".trim.wav")
        if not trimmed.exists():
            edge = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-af",
                            f"{edge},areverse,{edge},areverse", str(trimmed)], check=True)
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(trimmed)]))
        out[key] = (trimmed, dur)
    return out


def srt_time(s: float) -> str:
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_timeline(audio):
    timeline, t0 = [], 0.0
    for name, fn, base, vos, speech in SCENES:
        dur = base
        for key, off in vos:
            dur = max(dur, off + audio[key][1] + 0.45)
        timeline.append(dict(name=name, fn=fn, start=t0, dur=dur, vos=vos, speech=speech, st={}))
        t0 += dur
    return timeline, t0


def caption_for(scene, local, audio) -> tuple[str, float]:
    for key, off in scene["vos"]:
        end = off + audio[key][1] + 0.25
        if off - 0.2 <= local <= end:
            a = min(ease((local - off + 0.2) / FADE), ease((end - local) / FADE))
            return LINES[key], a
    return "", 0.0


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--voice", default="marin")
    ap.add_argument("--no-join", action="store_true", help="skip joining with the intro")
    args = ap.parse_args()

    audio = synth(args.voice)
    timeline, total = build_timeline(audio)
    end_dur = audio["end"][1]
    BULLET_STARTS[:] = [0.5, 0.5 + end_dur * 0.33, 0.5 + end_dur * 0.62]

    if args.preview:
        pv = HERE / "preview"
        pv.mkdir(exist_ok=True)
        picks = {"c1a": [1.0, C1A_LEAD + 4.5, C1A_LEAD + 14.0], "bridge": [1.5], "c1b": [C1B_LEAD + 4.0],
                 "c2": [1.8, c2_onset_t(LATENCY["mishra"]) + 0.4,
                       c2_onset_t(LATENCY["turn"]) + 0.4,
                       c2_onset_t(LATENCY["story"]) + 0.6], "montage": [6.5], "title": [3.0]}
        for s in timeline:
            for k, lt in enumerate(picks.get(s["name"], [s["dur"] * 0.8])):
                fr = s["fn"](lt, s["dur"], {})
                text, a = caption_for(s, lt, audio)
                draw_caption(fr, text, a)
                fr.convert("RGB").save(pv / f"{s['name']}_{k}.png")
        print(f"preview written ({total:.1f} s)")
        return

    silent = HERE / "cases_silent.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)],
        stdin=subprocess.PIPE,
    )
    n = round(total * FPS)
    last: dict[int, Image.Image] = {}
    for i in range(n):
        t = i / FPS
        idx = max(j for j, s in enumerate(timeline) if s["start"] <= t + 1e-9)
        s = timeline[idx]
        local = t - s["start"]
        frame = s["fn"](local, s["dur"], s["st"])
        text, a = caption_for(s, local, audio)
        draw_caption(frame, text, a)
        frame = frame.convert("RGB")
        if local < FADE and idx > 0 and s["name"] not in NO_CROSSFADE:
            frame = Image.blend(last[idx - 1], frame, ease(local / FADE))
        last[idx] = frame
        ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()

    with tempfile.TemporaryDirectory() as td:
        speech = Path(td) / "speech.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(OURS / "video.mp4"), "-vn",
                        "-ac", "2", "-ar", "48000", str(speech)], check=True)
        inputs, filters, k = [], [], 0
        for s in timeline:
            if s["speech"]:
                off, end = s["speech"]
                trim = f"atrim=0:{end},afade=t=out:st={end - 0.35}:d=0.35," if end else ""
                ms = round((s["start"] + off) * 1000)
                inputs += ["-i", str(speech)]
                filters.append(f"[{k + 1}:a]{trim}aresample=48000,adelay={ms}|{ms}[a{k}]")
                k += 1
            for key, off in s["vos"]:
                ms = round((s["start"] + off) * 1000)
                inputs += ["-i", str(audio[key][0])]
                filters.append(f"[{k + 1}:a]aresample=48000,aformat=channel_layouts=stereo,"
                               f"adelay={ms}|{ms}[a{k}]")
                k += 1
        mix = "".join(f"[a{j}]" for j in range(k))
        filters.append(f"{mix}amix=inputs={k}:normalize=0,apad[aout]")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs,
             "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest",
             str(HERE / "cases_vo.mp4")],
            check=True,
        )

    entries = []
    for s in timeline:
        for key, off in s["vos"]:
            s0 = s["start"] + off
            entries.append((s0, s0 + audio[key][1] + 0.2, LINES[key]))
    with open(HERE / "cases.srt", "w", encoding="utf-8") as f:
        for k, (a0, a1, text) in enumerate(entries, 1):
            f.write(f"{k}\n{srt_time(a0)} --> {srt_time(a1)}\n{text}\n\n")
    print(f"cases: {total:.1f} s, {n} frames")

    if not args.no_join:
        join(entries)


def join(case_entries) -> None:
    intro = VIDEO / "intro" / "intro_vo.mp4"
    cases = HERE / "cases_vo.mp4"
    out = VIDEO / "promo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(intro), "-i", str(cases), "-filter_complex",
         "[0:v]fps=30,format=yuv420p[v0];[1:v]fps=30,format=yuv420p[v1];"
         "[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
         "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
         "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "21",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
        check=True,
    )
    intro_len = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(intro)]))
    blocks = (VIDEO / "intro" / "intro.srt").read_text(encoding="utf-8").strip().split("\n\n")
    with open(VIDEO / "promo.srt", "w", encoding="utf-8") as f:
        k = 0
        for b in blocks:
            k += 1
            lines = b.split("\n")
            f.write(f"{k}\n" + "\n".join(lines[1:]) + "\n\n")
        for a0, a1, text in case_entries:
            k += 1
            f.write(f"{k}\n{srt_time(intro_len + a0)} --> {srt_time(intro_len + a1)}\n{text}\n\n")
    size = out.stat().st_size / 1e6
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)]))
    print(f"promo: {dur:.1f} s, {size:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
