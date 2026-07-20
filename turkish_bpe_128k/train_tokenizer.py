"""Train the 128K Turkish byte-level BPE tokeniser, GPT-4 / Llama-3 style.

Design choices, and why:

  * **Byte-level BPE.** The base alphabet is the 256 raw bytes, so *every*
    possible input is representable and encoding is lossless for any script --
    emoji, Cyrillic, CJK included. This is what every current frontier
    tokeniser does, and it is why none of them carry a live ``<unk>``: an
    unknown token would be unreachable code. Special tokens are still declared
    for downstream use (padding, sequence boundaries, masking).
  * **A Turkish-adapted GPT-4 split pattern.** The published cl100k / Llama-3
    regex opens with an English contraction clause (``'s|'t|'re|'ve|'m|'ll|'d``)
    that is actively harmful here: it would cut ``İstanbul'da`` into
    ``İstanbul`` + ``'d`` + ``a``, severing the apostrophe suffix Turkish uses
    on proper nouns. Dropping that clause lets ``[^\\r\\n\\p{L}\\p{N}]?\\p{L}+``
    keep ``'da`` together as one chunk. The rest of the pattern is unchanged.
  * **Digits in groups of up to three**, straight from cl100k -- it stops the
    vocabulary from burning merges on specific prices and years.
  * **No normaliser.** Frontier tokenisers avoid even NFC here, because any
    normalisation is technically lossy; byte-level BPE preserves the input
    exactly as given.

Run:  python turkish_bpe_128k/train_tokenizer.py
Out:  turkish_bpe_128k/tokenizer/  (tokenizer.json + transformers config files)
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent
CORPUS_FILE = HERE / "corpus" / "turkish_corpus.txt"
OUT_DIR = HERE / "tokenizer"

VOCAB_SIZE = 128_000
MIN_FREQUENCY = 2

# The cl100k_base pattern with the English contraction alternation removed.
SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?\p{L}+"     # optional leading symbol/space + a word
    r"|\p{N}{1,3}"                   # digits, at most three at a time
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"    # punctuation runs
    r"|\s*[\r\n]+"                   # line breaks
    r"|\s+(?!\S)"                    # trailing whitespace
    r"|\s+"                          # any remaining whitespace
)

# Reserved for downstream training; none of them fire during plain encoding.
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|mask|>"]


def build_tokenizer() -> Tokenizer:
    # unk_token=None: with 256 bytes as the alphabet nothing can be unknown.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),
        # use_regex=False: the Split above already did the chunking, so ByteLevel
        # only maps bytes to their printable stand-ins (space -> "Ġ", etc).
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def main() -> None:
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, always
        show_progress=True,
    )
    print(f"Training byte-level BPE (target vocab {VOCAB_SIZE:,}) on {CORPUS_FILE.name} ...")
    tokenizer.train([str(CORPUS_FILE)], trainer)

    reached = tokenizer.get_vocab_size()
    print(f"Reached {reached:,} tokens (target {VOCAB_SIZE:,})")
    if reached < VOCAB_SIZE:
        print("  NOTE: vocabulary plateaued below target -- the corpus ran out of"
              " pairs occurring at least twice.")

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|pad|>",
        mask_token="<|mask|>",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(OUT_DIR))
    print("Saved to", OUT_DIR.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
