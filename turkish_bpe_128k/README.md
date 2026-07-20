# turkish-bpe-128k

A **128,000-token byte-level BPE tokeniser for Turkish**, trained from scratch on
~200M characters of Turkish user-generated text with Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers) and packaged for
`transformers`.

Published at **[Erenyanic/turkish-bpe-128k](https://huggingface.co/Erenyanic/turkish-bpe-128k)**.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Erenyanic/turkish-bpe-128k")
ids = tok.encode("İstanbul'da yağmur yağıyor.", add_special_tokens=False)

# tokenize() returns byte-level forms -- see the note on `Ġ` and `Ã§` below.
tok.tokenize("İstanbul'da yağmur yağıyor.")
# ['Ä°stanbul', "'da", 'ĠyaÄŁmur', 'ĠyaÄŁÄ±yor', '.']

# Decode each id on its own to see the text the pieces stand for.
[tok.decode([i]) for i in ids]
# ['İstanbul', "'da", ' yağmur', ' yağıyor', '.']
```

This is a **separate sub-project** from the Middle-earth name generator in
[`src/`](../src/). It shares nothing with it but the repository — different data,
different tokeniser design, different scale.

## Design

The tokeniser follows what current frontier models (GPT-4, Llama 3, Qwen 3,
DeepSeek-V3) actually do, rather than the textbook BPE recipe.

| Choice | Value | Why |
| --- | --- | --- |
| Algorithm | Byte-level BPE | The 256 raw bytes are the base alphabet, so **any** input is representable |
| Vocab | 128,000 | In the modern band (Llama 3 = 128k, Qwen 3 ≈ 151k) |
| Normaliser | none | Even NFC is technically lossy; byte-level preserves input exactly |
| Casing | preserved | Keeps `Ankara` ≠ `ankara`, and sidesteps the Turkish dotted/dotless *i* trap |
| Digits | groups of ≤ 3 | Stops the vocab memorising specific prices and years |
| Unknown token | **none** | See below |

### Why there is no `<unk>`

Because with byte-level BPE it could never fire. Every input decomposes into one
of 256 byte tokens, all of which are in the vocabulary, so nothing is ever
out-of-vocabulary. An `<unk>` here would be unreachable code.

This is the reason the token has disappeared from frontier tokenisers. A live
`<unk>` is silent data destruction — the model cannot learn what it never sees,
and the loss surfaces only in production. Byte-level made that failure mode cost
nothing, so there was no reason to keep it.

The trade is real, though: a classic BPE with a capped alphabet gives you a
*meaningful* `<unk>` but **cannot round-trip** text containing characters outside
that alphabet. That data is simply gone. Byte-level trades the diagnostic signal
for guaranteed losslessness, which is the better deal for anything that will
front an actual model.

### The Turkish apostrophe fix

The published cl100k / Llama-3 split regex opens with an English contraction
clause — `'s|'t|'re|'ve|'m|'ll|'d`. Applied to Turkish it is a bug: `İstanbul'da`
would split into `İstanbul` + `'d` + `a`, severing the apostrophe suffix Turkish
uses on proper nouns. [`train_tokenizer.py`](train_tokenizer.py) drops that
clause, so `[^\r\n\p{L}\p{N}]?\p{L}+` keeps `'da` intact as one chunk. The rest
of the pattern is unchanged.

### The `Ġ` and `Ã§` you will see in the vocabulary

Byte-level BPE stores the 256 bytes remapped to printable Unicode, because raw
bytes include control characters that cannot be written safely into JSON. So
byte `0x20` (space) is stored as `Ġ`, and `ç` — two bytes in UTF-8 — appears as
`Ã§`. `tokenizer.json` therefore starts with a few special tokens, then those 256
byte tokens, then ~127,700 learned merges. Decoding turns them back into text;
nothing is wrong.

## Data

| Dataset | Column used | Lines kept |
| --- | --- | --- |
| [`winvoker/turkish-sentiment-analysis-dataset`](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) | `text` (labels discarded) | 487,070 |
| [`kmkarakaya/turkishReviews-ds`](https://huggingface.co/datasets/kmkarakaya/turkishReviews-ds) | `review` | 398,832 |
| **Total** | | **885,902 lines / 199.6M chars** |

All splits of both datasets are used — a tokeniser has no train/test leakage
concern, and holding data back would only make the vocabulary worse.

Cleaning is deliberately light ([`build_corpus.py`](build_corpus.py)): whitespace
collapsed to single spaces, records under 10 characters dropped, exact duplicates
removed. No lower-casing, no punctuation stripping, no accent folding — a
tokeniser should see text the way a model later will.

## Results

Measured on a 20,000-line random sample ([`evaluate.py`](evaluate.py)):

| Metric | Value |
| --- | --- |
| Vocabulary reached | 128,000 / 128,000 |
| Corpus round-trip | **20,000 / 20,000 exact** |
| Unseen-script round-trip | **PASS** (emoji, CJK, Cyrillic, Arabic) |
| Fertility | **1.239 tokens/word** (5.76 chars/token) |

Turkish is agglutinative, so segmentation quality is best judged on long suffix
chains:

```
evlerimizden          -> ['ev', 'lerimizden']
İstanbul'da yağmur    -> ['İstanbul', "'da", ' yağmur']
Çekoslovakyalılaştıramadıklarımızdan
                      -> ['Çek', 'os', 'lovak', 'yalı', 'laştır', 'amadık', 'larımızdan']
```

Roots separate from suffix chains rather than collapsing into opaque blobs,
which is the behaviour that matters for a Turkish model.

### Honest caveats

- **Fertility is flattered by the domain.** Both datasets are product and service
  reviews. 1.239 tokens/word partly reflects the tokeniser memorising in-domain
  vocabulary; on Turkish news, legal text or literature it will be worse. Treat
  it as an upper bound, not a general figure.
- **200M characters is modest for a 128K vocab.** For scale, GPT-2 drew a 50K
  vocab from ~40GB. The vocabulary did reach 128,000 and the rarest merges are
  still sane sub-words (`pişirme`, `pandemi`, `seti`) rather than memorised
  phrases — so the tail is not junk — but a broader corpus would produce a
  better-balanced vocabulary.
- **Single-domain register.** No news, wiki, literary or conversational text, so
  formal and archaic Turkish are under-represented.

## Reproduce

```bash
uv pip install tokenizers transformers datasets

python turkish_bpe_128k/build_corpus.py      # -> corpus/turkish_corpus.txt (~200MB)
python turkish_bpe_128k/train_tokenizer.py   # -> tokenizer/  (~25s)
python turkish_bpe_128k/evaluate.py          # self-test + report card
```

The corpus file is git-ignored; rebuild it with `build_corpus.py`.

## Files

```
turkish_bpe_128k/
├── build_corpus.py      # 1. merge both HF datasets into one text file
├── train_tokenizer.py   # 2. train the 128K byte-level BPE
├── evaluate.py          # 3. round-trip / coverage / fertility self-test
├── corpus/              #    generated, git-ignored
└── tokenizer/           #    tokenizer.json + transformers config
```
