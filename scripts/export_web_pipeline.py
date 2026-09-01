#!/usr/bin/env python3
"""Export Qwen2.5-0.5B EN pole-50 axes + scaler for the browser demo.

Loads only the tied embedding matrix from safetensors (no full model / MPS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dimension_extractor import DimensionExtractor  # noqa: E402

MODEL_PATH = ROOT / "models" / "Qwen" / "Qwen2.5-0.5B-Instruct"
POLE_REPORT = ROOT / "RAL" / "experiment" / "A" / "A1" / "result" / "en" / "pole_50" / "report.json"
OUT_PATH = ROOT / "web" / "public" / "pipeline" / "pipeline.json"
WEIGHTS = MODEL_PATH / "model.safetensors"


def load_embedding() -> torch.Tensor:
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for candidate in (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.model.embed_tokens.weight",
        ):
            if candidate in handle.keys():
                return handle.get_tensor(candidate).float()
        emb_keys = [k for k in keys if "embed" in k.lower() and k.endswith("weight")]
        if not emb_keys:
            raise KeyError(f"No embedding tensor in {WEIGHTS}; keys={keys[:20]}")
        return handle.get_tensor(emb_keys[0]).float()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Qwen checkpoint missing: {MODEL_PATH}")
    report = json.loads(POLE_REPORT.read_text(encoding="utf-8"))
    word_groups = report["word_groups"]
    scaler = report["scaler"]

    print("Loading embedding matrix…", flush=True)
    embedding = load_embedding()
    print(f"Embedding shape: {tuple(embedding.shape)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    extractor = DimensionExtractor(embedding, tokenizer, device="cpu")
    dimensions = extractor.extract_all_dimensions(
        {"word_groups": word_groups},
        dimensions=["valence", "arousal", "dominance"],
    )

    axes = {name: vec.float().cpu().tolist() for name, vec in dimensions.items()}
    hidden_size = len(next(iter(axes.values())))

    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "onnx_model": "onnx-community/Qwen2.5-0.5B-Instruct",
        "pole_n": int(report.get("pole_n", 50)),
        "language": "en",
        "hidden_size": hidden_size,
        "dimensions": ["valence", "arousal", "dominance"],
        "axes": axes,
        "scaler": {
            dim: {"scale": float(scaler[dim]["scale"]), "bias": float(scaler[dim]["bias"])}
            for dim in ["valence", "arousal", "dominance"]
        },
        "source": "RAL/experiment/A/A1/result/en/pole_50/report.json",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, hidden={hidden_size})", flush=True)


if __name__ == "__main__":
    main()
