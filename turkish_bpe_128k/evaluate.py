"""Self-test and report card for the Turkish 128K byte-level BPE tokeniser.

For a byte-level tokeniser the headline property is **total coverage**: every
input, in any script, must survive encode -> decode unchanged. So the tests are
the mirror image of what a classic ``<unk>`` tokeniser would need:

  1. **Round-trip on corpus text** -- must be 100%, no tolerated mismatches.
  2. **Round-trip on scripts absent from the training data** -- emoji, CJK,
     Cyrillic, Arabic. These are the discriminating cases: a classic BPE
     destroys them, byte-level must return them byte-for-byte.
  3. **Fertility** (tokens per word) on Turkish, plus a look at the merge tail,
     which is where a corpus too small for the vocab shows itself.

Run:  python turkish_bpe_128k/evaluate.py
"""

from __future__ import annotations

import random
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CORPUS_FILE = HERE / "corpus" / "turkish_corpus.txt"
OUT_DIR = HERE / "tokenizer"

SAMPLE_LINES = 20_000

# Turkish is agglutinative: one root grows a long suffix chain. A good tokeniser
# segments these into root + suffix pieces rather than one opaque blob.
AGGLUTINATION_DEMO = [
    "evlerimizden",
    "Çekoslovakyalılaştıramadıklarımızdan",
    "İstanbul'da yağmur yağıyor.",
    "Kitabı okudum ve çok beğendim.",
]

# None of these appear in the training corpus. All must round-trip exactly.
COVERAGE_DEMO = [
    "🎉 harika ürün 👍",
    "日本語のテキスト",
    "Привет, мир",
    "مرحبا بالعالم",
    "mixed: İstanbul 🇹🇷 東京 2024",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(OUT_DIR))
    print(f"vocab size: {len(tok):,}")
    print(f"specials:   {tok.all_special_tokens}\n")

    lines = CORPUS_FILE.read_text(encoding="utf-8").splitlines()
    random.seed(0)
    sample = random.sample(lines, min(SAMPLE_LINES, len(lines)))

    # --- 1. round-trip on corpus text ----------------------------------------
    failures = []
    total_tokens = total_words = total_chars = 0
    for line in sample:
        ids = tok.encode(line, add_special_tokens=False)
        if tok.decode(ids) != line:
            failures.append(line)
        total_tokens += len(ids)
        total_words += len(line.split())
        total_chars += len(line)

    status = "PASS" if not failures else f"FAIL ({len(failures)} mismatches)"
    print(f"[{status}] corpus round-trip: "
          f"{len(sample) - len(failures):,}/{len(sample):,} lines exact")

    # --- 2. round-trip on unseen scripts -------------------------------------
    print("\nunseen-script coverage (byte-level must return these unchanged):")
    all_ok = True
    for text in COVERAGE_DEMO:
        ids = tok.encode(text, add_special_tokens=False)
        ok = tok.decode(ids) == text
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}  ({len(ids)} tokens)")
    print(f"  -> {'all scripts round-trip losslessly' if all_ok else 'COVERAGE BROKEN'}")

    # --- 3. fertility and merge quality --------------------------------------
    print(f"\nfertility: {total_tokens / total_words:.3f} tokens/word,"
          f" {total_chars / total_tokens:.3f} chars/token")

    # tokenize() returns tokens in byte-level form, where a Turkish letter shows
    # as its UTF-8 bytes ("ç" -> "Ã§"). Decoding each id one at a time renders
    # the pieces as the text they actually stand for.
    print("\nsegmentation of agglutinated forms:")
    for text in AGGLUTINATION_DEMO:
        ids = tok.encode(text, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        print(f"  {text}")
        print(f"    -> {pieces}")

    # The last-learned merges are the rarest. If they are long whole phrases,
    # the corpus was small for the vocab and the tail is domain-fitted.
    vocab = tok.get_vocab()
    tail = sorted(vocab.items(), key=lambda kv: -kv[1])[:12]
    print("\nlast-learned (rarest) merges -- a small corpus shows up here:")
    print("  " + ", ".join(repr(tok.decode([i])) for _, i in reversed(tail)))


if __name__ == "__main__":
    main()
