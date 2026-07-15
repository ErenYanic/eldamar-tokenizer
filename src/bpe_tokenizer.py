"""Character-level BPE tokeniser for the Middle-earth names corpus.

We use Hugging Face's ``tokenizers`` library (the Rust-backed industry standard)
rather than hand-rolling BPE. The tokeniser is deliberately *character* level:

  * No ByteLevel pre-tokeniser, so the base alphabet is our ~42 real characters
    (each diacritic stays a single base token) instead of 256 raw bytes. That
    keeps both vocab targets -- 256 and 512 -- meaningful merge counts rather
    than making "vocab 256" a degenerate zero-merge tokeniser.
  * A ``Split("\n", isolated)`` pre-tokeniser makes every name its own unit and
    every newline a lone token, so merges never cross a name boundary and "\n"
    survives as the start/end-of-name marker (EOS) the models depend on.

The ``BpeTokenizer`` wrapper mirrors the repo's ``CharTokenizer`` interface
(``encode`` / ``decode`` / ``vocab_size`` / ``newline_id`` / ``eos_id``) so the
four model folders need almost no change to train on it.

Run:  python src/bpe_tokenizer.py          # trains vocab 256 and 512, self-tests
Out:  bpe/bpe_256.json, bpe/bpe_512.json
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
NAMES_FILE = ROOT / "data" / "middle_earth_names.txt"
BPE_DIR = ROOT / "bpe"

NEWLINE = "\n"
VOCAB_SIZES = (256, 512)


def build_bpe(names_path: Path, vocab_size: int) -> "BpeTokenizer":
    """Train a fresh character-level BPE tokeniser on a one-name-per-line file."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # Isolate newlines: each name becomes one pre-token, each "\n" its own -- so
    # the trainer only ever merges character pairs *within* a single name.
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=NEWLINE, behavior="isolated")
    # A no-op decoder; our wrapper reconstructs text by concatenating token
    # strings directly (the tokens are literal substrings, so this is exact).
    tokenizer.decoder = decoders.Fuse()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,          # allow even rare pairs, so small data can grow the vocab
        show_progress=False,
        special_tokens=[],        # "\n" is a normal character token, not a special one
    )
    tokenizer.train([str(names_path)], trainer)
    return BpeTokenizer(tokenizer)


class BpeTokenizer:
    """Thin adapter around a trained ``tokenizers.Tokenizer`` mirroring CharTokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        newline_id = tokenizer.token_to_id(NEWLINE)
        if newline_id is None:
            raise ValueError("The trained vocabulary has no newline token.")
        # The newline both separates names and marks end-of-sequence (EOS).
        self.newline_id = newline_id
        self.eos_id = newline_id

    @classmethod
    def from_file(cls, path: str | Path) -> "BpeTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        # Concatenate the literal token strings -- exact for character-level BPE,
        # and avoids the space-joining that Tokenizer.decode applies by default.
        return "".join(self._tok.id_to_token(i) for i in ids)


def main() -> None:
    text = NAMES_FILE.read_text(encoding="utf-8")
    names = [n for n in text.split(NEWLINE) if n]
    base_alphabet = sorted(set(text) - {NEWLINE})
    print(f"Corpus: {len(names)} names, {len(text)} chars, "
          f"{len(base_alphabet)} base letters (+ newline)\n")

    for vocab_size in VOCAB_SIZES:
        tok = build_bpe(NAMES_FILE, vocab_size)
        save_path = BPE_DIR / f"bpe_{vocab_size}.json"
        tok.save(save_path)

        # Round-trip the whole corpus to prove encode/decode is lossless.
        ids = tok.encode(text)
        assert tok.decode(ids) == text, "round-trip mismatch!"

        avg_tokens = sum(len(tok.encode(n)) for n in names) / len(names)
        print(f"vocab {vocab_size}: reached {tok.vocab_size} tokens "
              f"(target {vocab_size}), newline_id={tok.newline_id}")
        print(f"  corpus encodes to {len(ids)} ids; {avg_tokens:.2f} tokens/name")
        for sample in ("isildur", "gondor", "galadriel", "éowyn"):
            pieces = [tok._tok.id_to_token(i) for i in tok.encode(sample)]
            print(f"  {sample:<11} -> {pieces}")
        print(f"  saved {save_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
