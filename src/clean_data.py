"""Turn the raw Middle-earth sources into one clean name-per-line corpus.

Inputs (all under data/):
  * Characters.csv        -- column "Name"
  * lotr_characters.csv   -- column "name"
  * characters_data.csv   -- column "name"
  * locations.txt         -- one raw place name per line (from scrape_locations.py)

Every source is passed through the *same* normalising pipeline so characters and
places are treated identically:

  1. Normalise unicode (NFC) and drop invisible junk (soft hyphens, zero-widths).
     Diacritics themselves are kept -- they are half the Elvish flavour.
  2. Mine any parentheticals: strip them from the main string, then feed their
     contents back through the pipeline too. This keeps real alternative names
     -- "Belladonna (Took) Baggins" yields "took" -- while the stop-word and
     numeral filters below discard disambiguation prose like "(son of Axantur)".
  3. Lower-case with plain str.lower() (NOT Turkish lowering, which would turn
     "Isildur" into "ısildur").
  4. Split on whitespace so multi-word names become one word per line.
  5. Drop connective / descriptive stop-words, regnal Roman numerals, and any
     token that is not a real word. Intra-word hyphens and apostrophes survive
     ("aelin-uial"); a trailing possessive "'s" is trimmed.
  6. De-duplicate (the model learns a distribution; duplicates just re-weight it).

Run:  python src/clean_data.py
Out:  data/middle_earth_names.txt
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "middle_earth_names.txt"

# (filename, column). column=None means "one raw name per line" (locations.txt).
SOURCES: list[tuple[str, str | None]] = [
    ("Characters.csv", "Name"),
    ("lotr_characters.csv", "name"),
    ("characters_data.csv", "name"),
    ("locations.txt", None),
]

# Connective words and generic geographic descriptors: real text, but not names
# on their own, so they are removed once multi-word entries are split.
STOP_WORDS = frozenset(
    {
        # connectives / relations
        "of", "the", "a", "an", "and", "or", "at", "in", "on", "to", "by",
        "son", "daughter", "wife", "husband", "unnamed",
        # abbreviations and Sindarin grammatical particles left by splitting
        # multi-word names ("Tolman Cotton Jr.", "Bar-en-Danwedh", "Arthor na ...")
        "jr", "sr", "na", "nan", "en",
        # geographic common nouns
        "mount", "mountain", "mountains", "lake", "river", "hill", "hills",
        "pass", "forest", "vale", "valley", "isle", "island", "bay", "gulf",
        "cape", "tower", "gate", "city", "land", "sea", "road", "field",
        "fields", "wood", "woods", "gates", "towers",
    }
)

# Invisible characters that sometimes ride along inside wiki/CSV text.
INVISIBLE = {
    "­",  # soft hyphen
    "​", "‌", "‍",  # zero-width space / non-joiner / joiner
    "﻿",  # byte-order mark
}
INVISIBLE_TABLE = {ord(c): None for c in INVISIBLE}

PAREN_RE = re.compile(r"\(([^)]*)\)")
# Regnal numerals up to ~39 (Durin VII, Ecthelion II, ...); only i/v/x so real
# names made of other letters are never mistaken for numerals.
ROMAN_RE = re.compile(r"^(x{0,3})(ix|iv|v?i{0,3})$")
POSSESSIVE_RE = re.compile(r"['’]s$")


def normalise(text: str) -> str:
    """NFC-normalise, drop invisibles, and unify curly quotes / dashes."""
    text = unicodedata.normalize("NFC", text).translate(INVISIBLE_TABLE)
    text = text.replace("’", "'").replace("‘", "'")
    for dash in ("–", "—", "−"):  # en / em / minus -> hyphen
        text = text.replace(dash, "-")
    return text


def clean_word(word: str) -> str | None:
    """Reduce a single whitespace-delimited token to a valid name, or None."""
    # Trim leading/trailing punctuation, keeping internal hyphens/apostrophes.
    word = re.sub(r"^[\W_]+", "", word)
    word = re.sub(r"[\W_]+$", "", word)
    word = POSSESSIVE_RE.sub("", word)
    word = word.strip("-'")
    if len(word) < 2:
        return None
    if word in STOP_WORDS or ROMAN_RE.match(word):
        return None
    # Must read as a word: letters plus optional internal hyphen/apostrophe.
    if not all(ch.isalpha() or ch in "-'" for ch in word):
        return None
    if not any(ch.isalpha() for ch in word):
        return None
    return word


def clean_entry(raw: str) -> list[str]:
    """Expand one raw source entry into zero or more clean name tokens."""
    raw = normalise(raw).strip()
    if not raw:
        return []

    # Split the parenthetical contents off and treat them as extra material.
    chunks = [PAREN_RE.sub(" ", raw)]
    chunks.extend(PAREN_RE.findall(raw))

    words: list[str] = []
    for chunk in chunks:
        for token in chunk.lower().split():
            cleaned = clean_word(token)
            if cleaned:
                words.append(cleaned)
    return words


def read_source(filename: str, column: str | None) -> list[str]:
    """Yield the raw name strings from one source file."""
    path = DATA_DIR / filename
    text = path.read_text(encoding="utf-8")
    if column is None:
        return [line for line in text.splitlines() if line.strip()]
    reader = csv.DictReader(text.splitlines())
    return [row[column] for row in reader if row.get(column)]


def main() -> None:
    seen: set[str] = set()
    names: list[str] = []

    for filename, column in SOURCES:
        raw_rows = read_source(filename, column)
        added = 0
        for raw in raw_rows:
            for name in clean_entry(raw):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
                    added += 1
        print(f"  {filename:<22} {len(raw_rows):>5} rows -> {added:>4} new names")

    names.sort()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(names) + "\n", encoding="utf-8")

    alphabet = sorted({ch for name in names for ch in name})
    print(f"\nTotal unique names: {len(names)}")
    print(f"Alphabet ({len(alphabet)} chars): {''.join(alphabet)}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
