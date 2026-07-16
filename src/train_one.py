"""Train ONE (architecture, vocab size) pair and save its checkpoint.

This is the worker invoked once per model by train_all.py. It lives in its own
process on purpose: every architecture folder uses flat module names
(config.py, model.py, block.py, ...), so importing two of them into a single
interpreter would collide in sys.modules. One process = one architecture keeps
each import clean (the repo's "one architecture per kernel" rule).

The training recipe is the repo's own -- a plain windowed next-token loop -- on
the Middle-earth corpus. The tokeniser is selectable: our character-level BPE at
vocab 256 or 512, or the repo's plain CharTokenizer ("char") as a baseline.

Run:  python src/train_one.py qwen3 256
      python src/train_one.py qwen3 char
      python src/train_one.py deepseek3 512 --steps 200   # quick smoke test
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "single_letter_transformers"
DATA_FILE = ROOT / "data" / "middle_earth_names.txt"
BPE_DIR = ROOT / "bpe"
CKPT_DIR = ROOT / "checkpoints"

# architecture -> (folder, model class, default training steps).
# DeepSeek's sparse routing needs a little longer to settle, as in the repo.
ARCHITECTURES = {
    "qwen3": ("qwen3", "TinyQwen", 5000),
    "qwen3_5": ("qwen3_5", "TinyQwen35", 5000),
    "gemma4": ("gemma4", "TinyGemma", 5000),
    "deepseek3": ("deepseek3", "TinyDeepSeek", 5000),
}

# Shared hyperparameters (identical to the repo's train.py scripts).
BATCH_SIZE = 64
BLOCK_SIZE = 16
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337


def load_architecture(folder: str, class_name: str):
    """Import one architecture's ModelConfig + model class from its own folder."""
    sys.path.insert(0, str(REPO / folder))
    model_config = importlib.import_module("config").ModelConfig
    model_class = getattr(importlib.import_module("model"), class_name)
    return model_config, model_class


def build_tokenizer(kind: str):
    """Return (tokenizer, label, checkpoint-metadata) for 'char', '256' or '512'.

    Call this only after load_architecture(), because the 'char' baseline imports
    the CharTokenizer from the architecture folder now on sys.path. Both tokenisers
    expose the same interface (encode/decode/vocab_size/newline_id/eos_id).
    """
    if kind == "char":
        # The repo's plain character tokeniser; its vocabulary is built directly
        # from the corpus (42 letters + newline).
        char_tokenizer = importlib.import_module("tokenizer").CharTokenizer
        tokenizer = char_tokenizer.from_file(str(DATA_FILE))
        return tokenizer, "char", {"tokenizer_kind": "char", "chars": tokenizer.chars}

    from bpe_tokenizer import BpeTokenizer

    vocab = int(kind)
    tokenizer = BpeTokenizer.from_file(BPE_DIR / f"bpe_{vocab}.json")
    return tokenizer, f"bpe{vocab}", {"tokenizer_kind": "bpe", "tokenizer": f"bpe/bpe_{vocab}.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one tiny model on the names corpus.")
    parser.add_argument("arch", choices=sorted(ARCHITECTURES))
    parser.add_argument("tokenizer", choices=("char", "256", "512"),
                        help="'char' = plain CharTokenizer baseline; 256/512 = BPE vocab size")
    parser.add_argument("--steps", type=int, default=None, help="override the default step count")
    args = parser.parse_args()

    folder, class_name, default_steps = ARCHITECTURES[args.arch]
    steps = args.steps if args.steps is not None else default_steps

    model_config, model_class = load_architecture(folder, class_name)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, label, tok_meta = build_tokenizer(args.tokenizer)
    text = DATA_FILE.read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    cfg = model_config(vocab_size=tokenizer.vocab_size)
    model = model_class(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch} · {label}] device={device} "
          f"params={n_params:,} steps={steps} corpus_ids={len(data)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def get_batch():
        """Sample random windows; targets are inputs shifted by one token."""
        ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
        return x.to(device), y.to(device)

    def sample_names(n: int = 10, max_new_tokens: int = 20) -> list[str]:
        """Generate a few names, each starting from the newline (EOS) token."""
        model.eval()
        start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long, device=device)
        out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0,
                             top_k=None, eos_id=tokenizer.eos_id)
        model.train()
        return [tokenizer.decode(row[1:]).split("\n")[0] for row in out.tolist()]

    final_loss = float("nan")
    for step in range(1, steps + 1):
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"  step {step:5d}  loss {final_loss:.4f}")

    baseline = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
    print(f"  baseline loss (uniform guessing): {baseline:.4f}")
    print("  samples: " + ", ".join(sample_names(10)))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CKPT_DIR / f"{args.arch}_{label}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "arch": args.arch,
            "vocab_size": tokenizer.vocab_size,
            **tok_meta,
        },
        out_path,
    )
    print(f"  saved {out_path.relative_to(ROOT)}  (final loss {final_loss:.4f})")


if __name__ == "__main__":
    main()
