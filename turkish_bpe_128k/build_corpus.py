"""Build the plain-text training corpus for the Turkish 64K BPE tokeniser.

Two Hugging Face datasets are merged into one file, one text per line:

  * ``winvoker/turkish-sentiment-analysis-dataset`` -- product/film/social
    comments. Only the ``text`` column is used; the sentiment ``label`` and the
    ``dataset`` provenance column are irrelevant to tokeniser training.
  * ``kmkarakaya/turkishReviews-ds`` -- longer free-text customer reviews, in a
    ``review`` column.

Both are read across *all* their splits: a tokeniser has no train/test leakage
concern, so holding data back would only make the vocabulary worse.

Cleaning is deliberately light. A tokeniser should see text the way a model
later will, so we only normalise whitespace, drop empties and de-duplicate --
no lower-casing, no punctuation stripping, no accent folding.

Run:  python turkish_bpe_64k/build_corpus.py
Out:  turkish_bpe_64k/corpus/turkish_corpus.txt
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
CORPUS_FILE = CORPUS_DIR / "turkish_corpus.txt"

# (dataset id, column holding the text)
SOURCES = [
    ("winvoker/turkish-sentiment-analysis-dataset", "text"),
    ("kmkarakaya/turkishReviews-ds", "review"),
]

MIN_CHARS = 10          # drop fragments too short to teach a merge anything
WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str | None:
    """Normalise one record to a single line, or return None to drop it."""
    if not text:
        return None
    # NFC keeps the Turkish letters (ş ğ ı İ ö ü ç) as single code points, so the
    # BPE alphabet sees one symbol per letter rather than letter + combining mark.
    text = unicodedata.normalize("NFC", text)
    # Collapse every run of whitespace -- including the newlines inside multi-line
    # reviews -- to one space, so "one record per line" actually holds.
    text = WHITESPACE.sub(" ", text).strip()
    return text if len(text) >= MIN_CHARS else None


def main() -> None:
    seen: set[str] = set()
    lines: list[str] = []
    kept_per_source: dict[str, int] = {}

    for dataset_id, column in SOURCES:
        before = len(lines)
        dataset = load_dataset(dataset_id)
        for split in dataset:
            for text in dataset[split][column]:
                line = clean(text)
                if line is not None and line not in seen:
                    seen.add(line)
                    lines.append(line)
        kept_per_source[dataset_id] = len(lines) - before

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_chars = sum(len(line) for line in lines)
    print("Corpus written to", CORPUS_FILE.relative_to(HERE.parent))
    for dataset_id, kept in kept_per_source.items():
        print(f"  {kept:>8,} new lines from {dataset_id}")
    print(f"  {len(lines):>8,} lines total")
    print(f"  {total_chars:>8,} characters ({total_chars / 1e6:.1f}M)")
    print(f"  {len(set(''.join(lines))):>8,} distinct characters")


if __name__ == "__main__":
    main()
