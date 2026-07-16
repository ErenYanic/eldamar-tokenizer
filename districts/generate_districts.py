"""Generate Turkish district (ilçe) names from a trained checkpoint.

Companion to train_districts.py. Reuses the repo's CharTokenizer and model class
from the architecture folder; one architecture per process.

Run:  python districts/generate_districts.py qwen3
      python districts/generate_districts.py gemma4 --count 30 --temperature 0.7
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "single_letter_transformers"
CKPT_DIR = ROOT / "districts" / "checkpoints"

ARCHITECTURES = {
    "qwen3": ("qwen3", "TinyQwen"),
    "qwen3_5": ("qwen3_5", "TinyQwen35"),
    "gemma4": ("gemma4", "TinyGemma"),
    "deepseek3": ("deepseek3", "TinyDeepSeek"),
}


def load(arch: str):
    folder, class_name = ARCHITECTURES[arch]
    sys.path.insert(0, str(REPO / folder))
    char_tokenizer = importlib.import_module("tokenizer").CharTokenizer
    model_class = getattr(importlib.import_module("model"), class_name)

    checkpoint_path = CKPT_DIR / f"{arch}.pt"
    if not checkpoint_path.exists():
        sys.exit(f"No checkpoint at {checkpoint_path.relative_to(ROOT)} -- train it first.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    tokenizer = char_tokenizer(ckpt["chars"])
    model = model_class(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_names(model, tokenizer, count: int, temperature: float) -> list[str]:
    start = torch.full((count, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=temperature, top_k=None, eos_id=tokenizer.eos_id)
    names = []
    for row in out.tolist():
        name = tokenizer.decode(row[1:]).split("\n")[0]
        if name:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample district names from a checkpoint.")
    parser.add_argument("arch", choices=sorted(ARCHITECTURES))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load(args.arch)
    for name in generate_names(model, tokenizer, args.count, args.temperature):
        print(name)


if __name__ == "__main__":
    main()
