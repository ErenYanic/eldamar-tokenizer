"""Generate Middle-earth names from any trained checkpoint.

Works for all four architectures: it reads the checkpoint's architecture folder,
rebuilds the model, and samples with the matching BPE tokeniser. Like the worker
in train_one.py it runs one architecture per process, so the flat module names in
the architecture folders never collide.

Run:  python src/generate.py qwen3 512                 # 20 names, temperature 0.8
      python src/generate.py gemma4 char --count 40    # the CharTokenizer baseline
      python src/generate.py deepseek3 512 --temperature 0.7 --novel-only

Lower temperature -> safer, more familiar names. Higher -> more varied/inventive.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "single_letter_transformers"
CKPT_DIR = ROOT / "checkpoints"
NAMES_FILE = ROOT / "data" / "middle_earth_names.txt"

# architecture -> (folder, model class name).
ARCHITECTURES = {
    "qwen3": ("qwen3", "TinyQwen"),
    "qwen3_5": ("qwen3_5", "TinyQwen35"),
    "gemma4": ("gemma4", "TinyGemma"),
    "deepseek3": ("deepseek3", "TinyDeepSeek"),
}


def load_model_and_tokenizer(arch: str, tok_spec: str):
    """Rebuild the trained model and its tokeniser (char or BPE) from the checkpoint."""
    folder, class_name = ARCHITECTURES[arch]
    sys.path.insert(0, str(REPO / folder))
    # The architecture folder must be importable *before* torch.load, because the
    # pickled cfg is that folder's config.ModelConfig.
    model_class = getattr(importlib.import_module("model"), class_name)

    name = "char" if tok_spec == "char" else f"bpe{tok_spec}"
    checkpoint_path = CKPT_DIR / f"{arch}_{name}.pt"
    if not checkpoint_path.exists():
        sys.exit(f"No checkpoint at {checkpoint_path.relative_to(ROOT)} -- train it first.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = model_class(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    if ckpt.get("tokenizer_kind") == "char":
        # Rebuild the CharTokenizer from the exact character list it was trained on.
        char_tokenizer = importlib.import_module("tokenizer").CharTokenizer
        tokenizer = char_tokenizer(ckpt["chars"])
    else:
        from bpe_tokenizer import BpeTokenizer
        tokenizer = BpeTokenizer.from_file(ROOT / ckpt["tokenizer"])
    return model, tokenizer


@torch.no_grad()
def generate_names(model, tokenizer, count: int, temperature: float) -> list[str]:
    """Sample `count` names, each starting from the newline (start-of-name) token."""
    start = torch.full((count, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=temperature, top_k=None, eos_id=tokenizer.eos_id)
    names = []
    for row in out.tolist():
        # Drop the leading newline, then keep everything up to the next newline.
        name = tokenizer.decode(row[1:]).split("\n")[0]
        if name:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample names from a trained checkpoint.")
    parser.add_argument("arch", choices=sorted(ARCHITECTURES))
    parser.add_argument("tokenizer", choices=("char", "256", "512"),
                        help="'char' for the CharTokenizer baseline, or a BPE vocab size")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None, help="fix the RNG for reproducible samples")
    parser.add_argument("--novel-only", action="store_true",
                        help="only show names that are NOT already in the training corpus")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.arch, args.tokenizer)
    names = generate_names(model, tokenizer, args.count, args.temperature)

    if args.novel_only:
        known = set(NAMES_FILE.read_text(encoding="utf-8").split("\n"))
        names = [n for n in names if n not in known]

    for name in names:
        print(name)


if __name__ == "__main__":
    main()
