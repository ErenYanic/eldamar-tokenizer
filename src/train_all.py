"""Train every (architecture x vocab size) model: 4 x 2 = 8 checkpoints.

Each model is trained by launching train_one.py in a fresh subprocess, because
the architecture folders share flat module names and cannot coexist in one
interpreter. The worker streams its own progress; we just orchestrate and
report a final pass/fail summary.

Run:  python src/train_all.py                 # full sweep, all 8 models
      python src/train_all.py --steps 200     # quick smoke run of all 8
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "train_one.py"

ARCHITECTURES = ("qwen3", "qwen3_5", "gemma4", "deepseek3")
VOCAB_SIZES = (256, 512)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all 8 tiny models on the BPE corpus.")
    parser.add_argument("--steps", type=int, default=None, help="override step count for every run")
    args = parser.parse_args()

    runs = list(itertools.product(ARCHITECTURES, VOCAB_SIZES))
    results: list[tuple[str, int, int]] = []

    for index, (arch, vocab) in enumerate(runs, start=1):
        print(f"\n{'=' * 60}\n[{index}/{len(runs)}] training {arch} · vocab {vocab}\n{'=' * 60}")
        command = [sys.executable, str(WORKER), arch, str(vocab)]
        if args.steps is not None:
            command += ["--steps", str(args.steps)]
        completed = subprocess.run(command, cwd=str(ROOT))
        results.append((arch, vocab, completed.returncode))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for arch, vocab, code in results:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {arch:<10} vocab {vocab}: {status}")

    if any(code != 0 for _, _, code in results):
        sys.exit("Some training runs failed.")


if __name__ == "__main__":
    main()
