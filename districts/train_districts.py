"""Train ONE architecture to generate Turkish district (ilçe) names.

Separate side task, kept apart from the eldamar-tokenizer BPE pipeline. It reuses
the ORIGINAL repo's pieces unchanged: the CharTokenizer from each architecture's
tokenizer.py, and the model/config from that same folder. The corpus is the
district list cleaned by the repo's own temizle_isimler.py.

One architecture per process (the folders share flat module names), so run it once
per model.

Run:  python districts/train_districts.py qwen3
      python districts/train_districts.py deepseek3
Out:  districts/checkpoints/<arch>.pt
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "single_letter_transformers"
DATA_FILE = ROOT / "districts" / "ilceler_temiz.txt"
CKPT_DIR = ROOT / "districts" / "checkpoints"

# architecture -> (folder, model class, steps) -- steps mirror the repo's own train.py.
ARCHITECTURES = {
    "qwen3": ("qwen3", "TinyQwen", 5000),
    "qwen3_5": ("qwen3_5", "TinyQwen35", 5000),
    "gemma4": ("gemma4", "TinyGemma", 5000),
    "deepseek3": ("deepseek3", "TinyDeepSeek", 5000),
}

BATCH_SIZE = 64
BLOCK_SIZE = 16
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one tiny model on Turkish district names.")
    parser.add_argument("arch", choices=sorted(ARCHITECTURES))
    parser.add_argument("--steps", type=int, default=None, help="override the default step count")
    args = parser.parse_args()

    folder, class_name, default_steps = ARCHITECTURES[args.arch]
    steps = args.steps if args.steps is not None else default_steps

    # Reuse the repo's CharTokenizer, ModelConfig and model class from the folder.
    sys.path.insert(0, str(REPO / folder))
    char_tokenizer = importlib.import_module("tokenizer").CharTokenizer
    model_config = importlib.import_module("config").ModelConfig
    model_class = getattr(importlib.import_module("model"), class_name)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = char_tokenizer.from_file(str(DATA_FILE))
    text = DATA_FILE.read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    cfg = model_config(vocab_size=tokenizer.vocab_size)
    model = model_class(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch}] device={device} vocab_size={tokenizer.vocab_size} "
          f"params={n_params:,} steps={steps} corpus_chars={len(data)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def get_batch():
        ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
        return x.to(device), y.to(device)

    def sample_names(n: int = 10, max_new_tokens: int = 20) -> list[str]:
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
    out_path = CKPT_DIR / f"{args.arch}.pt"
    torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "cfg": cfg}, out_path)
    print(f"  saved {out_path.relative_to(ROOT)}  (final loss {final_loss:.4f})")


if __name__ == "__main__":
    main()
