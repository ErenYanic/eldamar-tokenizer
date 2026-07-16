"""Train every (architecture x tokeniser) model.

Tokenisers are 'char' (CharTokenizer baseline), 256 and 512 (BPE), so the full
sweep is 4 architectures x 3 tokenisers = 12 checkpoints. Use --tokenizer to
restrict the run (e.g. only the 4 char baselines).

Each model is trained by launching train_one.py in a fresh subprocess, because
the architecture folders share flat module names and cannot coexist in one
interpreter. The worker streams its own progress; we just orchestrate and
report a final pass/fail summary.

Run:  python src/train_all.py                     # full sweep, all 12 models
      python src/train_all.py --tokenizer char    # only the 4 char baselines
      python src/train_all.py --steps 200         # quick smoke run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "train_one.py"

ARCHITECTURES = ("qwen3", "qwen3_5", "gemma4", "deepseek3")
TOKENIZERS = ("char", "256", "512")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny models on the names corpus.")
    parser.add_argument("--tokenizer", choices=TOKENIZERS, default=None,
                        help="restrict to one tokeniser (default: all three)")
    parser.add_argument("--steps", type=int, default=None, help="override step count for every run")
    args = parser.parse_args()

    tokenizers = (args.tokenizer,) if args.tokenizer else TOKENIZERS
    runs = [(arch, tok) for arch in ARCHITECTURES for tok in tokenizers]
    results: list[tuple[str, str, int]] = []

    for index, (arch, tok) in enumerate(runs, start=1):
        print(f"\n{'=' * 60}\n[{index}/{len(runs)}] training {arch} · {tok}\n{'=' * 60}")
        command = [sys.executable, str(WORKER), arch, tok]
        if args.steps is not None:
            command += ["--steps", str(args.steps)]
        completed = subprocess.run(command, cwd=str(ROOT))
        results.append((arch, tok, completed.returncode))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for arch, tok, code in results:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {arch:<10} {tok:<4}: {status}")

    if any(code != 0 for _, _, code in results):
        sys.exit("Some training runs failed.")


if __name__ == "__main__":
    main()
