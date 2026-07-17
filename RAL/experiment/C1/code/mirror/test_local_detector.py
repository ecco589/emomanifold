#!/usr/bin/env python3
print("[1] start", flush=True)
from video_va_pipeline import make_detector
print("[2] make_detector...", flush=True)
d = make_detector("cpu")
print("[3] OK", d, flush=True)
