# eldamar-tokenizer

A tiny, end-to-end pipeline that **learns to invent Middle-earth names**. It
cleans a dataset of Tolkien character and place names, trains a character-level
**BPE** tokeniser at two vocabulary sizes (256 and 512), and re-trains four
tiny from-scratch LLM architectures on the result — **8 models in total**
(4 architectures × 2 vocab sizes).

```
$ python src/generate.py gemma4 512 --count 8 --temperature 0.8 --seed 1
shield    boromir    markhîr    pengolodh
brandyuis echoriath  arches     running
```

`boromir`, `pengolodh` and `echoriath` are real Tolkien names the model has
memorised; `markhîr` and `brandyuis` are invented but phonetically Tolkienish;
`shield` and `arches` are ordinary English words the dataset deliberately keeps
(they fall out of splitting multi-word place names). All were produced by models
small enough to train on a CPU in a couple of minutes.

## Credit / attribution

The four model architectures come from
[**malibayram/single_letter_transformers**](https://github.com/malibayram/single_letter_transformers),
a beautiful set of from-scratch tiny-LLM implementations (Qwen3, Qwen3.5, Gemma,
DeepSeek-V3). That repository lives unchanged in
[`single_letter_transformers/`](single_letter_transformers/) as plain content; all
credit for the model code belongs to its author. `eldamar-tokenizer` only swaps
the **data** (Turkish names → Middle-earth names) and the **tokeniser**
(character-level → character-level **BPE**), and adds the glue to run the sweep.

## What is different from the original

| | Original (`single_letter_transformers`) | This project |
| --- | --- | --- |
| Task | Generate Turkish first names | Generate Middle-earth names |
| Data | 921 cleaned Turkish names | 2,189 cleaned Tolkien names (characters + places) |
| Tokeniser | Character level (~30 tokens) | Character-level **BPE** (256 and 512) |
| Models | 1 per architecture | **2 per architecture** (one per vocab size) = 8 |

## Pipeline

```
3 CSVs  ─┐
         ├─► clean_data.py ─► data/middle_earth_names.txt ─► bpe_tokenizer.py ─► bpe/bpe_{256,512}.json
locations┘                          (2,189 names)                                        │
 (scraped)                                                                               ▼
                                                                     train_all.py ─► checkpoints/*.pt (8)
                                                                                         │
                                                                                         ▼
                                                                              generate.py ─► names
```

## Directory layout

```
eldamar-tokenizer/
├── data/
│   ├── Characters.csv, lotr_characters.csv, characters_data.csv   # raw name sources
│   ├── locations.txt              # place names scraped from Tolkien Gateway
│   └── middle_earth_names.txt     # the clean corpus, one name per line
├── bpe/
│   ├── bpe_256.json               # trained BPE tokenisers (Hugging Face format)
│   └── bpe_512.json
├── checkpoints/                   # 8 trained models: {arch}_bpe{256,512}.pt
├── src/
│   ├── scrape_locations.py        # 1. fetch place names via the MediaWiki API
│   ├── clean_data.py              # 2. merge + clean all sources into one corpus
│   ├── bpe_tokenizer.py           # 3. train / load the character-level BPE tokeniser
│   ├── train_one.py               # 4. train a single (arch, vocab) model
│   ├── train_all.py               #    ... orchestrate all 8
│   └── generate.py                # 5. sample names from any checkpoint
└── single_letter_transformers/    # the original repo (model code), unchanged
```

## Data cleaning

Three character CSVs plus a scraped list of place names all pass through the **same**
pipeline in [`src/clean_data.py`](src/clean_data.py):

- **Lower-case** with plain `str.lower()` (not Turkish lowering, which would turn
  `Isildur` into `ısildur`).
- **Keep diacritics** (`á â ä é ê ë í î ó ô ö ú û`) — they are half the Elvish flavour.
- **Mine parentheticals**: strip them from the main string but feed their contents
  back through the pipeline, so `Belladonna (Took) Baggins` yields `took` while
  filters discard disambiguation prose like `(son of Axantur)`.
- **Split** multi-word names into one word per line; keep internal hyphens
  (`aelin-uial`, `ar-pharazôn`).
- **Filter** connective/geographic stop-words (of, the, mount, river, …), regnal
  Roman numerals (II, VI, …) and abbreviations (Jr).
- **De-duplicate** and sort.

Result: **2,189 unique names**, a **42-character** alphabet (plus newline).

Place names come from Tolkien Gateway's
[`Index:Locations`](https://tolkiengateway.net/wiki/Index:Locations). The site blocks
plain fetches, so [`src/scrape_locations.py`](src/scrape_locations.py) uses its
MediaWiki API with a descriptive User-Agent and extracts the display text of every
`[[link]]` bullet.

## Tokeniser

[`src/bpe_tokenizer.py`](src/bpe_tokenizer.py) trains a **character-level** BPE with
Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers):

- **No byte-level pre-tokeniser**, so the base alphabet is the 42 real characters
  (each diacritic is one base token). This keeps both 256 and 512 meaningful merge
  targets rather than making "vocab 256" a degenerate zero-merge tokeniser.
- A `Split("\n", isolated)` pre-tokeniser means merges never cross a name boundary
  and `\n` stays a lone token — the start/end-of-name marker (EOS) the models rely on.
- Encoding is **lossless** (verified by a full-corpus round-trip).

| Vocab | Tokens reached | Tokens / name |
| ----- | -------------- | ------------- |
| 256   | 256            | 3.68          |
| 512   | 512            | 3.12          |

Example: `galadriel` → `gal·ad·ri·el` (256) → `gal·ad·riel` (512).

## Results

All eight models drop far below the uniform-guessing baseline (`ln 256 ≈ 5.55`,
`ln 512 ≈ 6.24`) after 5,000 steps on CPU.

| Architecture | Params (256 / 512) | Final loss (256 / 512) | Sample names |
| ------------ | ------------------ | ---------------------- | ------------ |
| **Qwen3** (dense) | 26.8k / 35.0k | 1.47 / 1.25 | siriondir, alcarnor, kingsland |
| **Qwen3.5** (hybrid) | 49.2k / 57.4k | 0.76 / 0.64 | elfstan, menelvy, aerinel |
| **Gemma** (sliding-window) | 113.5k / 170.9k | 0.70 / **0.56** | thorondor, entwash, clayhanger |
| **DeepSeek-V3** (MoE) | 55.2k / 63.4k | 0.94 / 0.82 | drúadan, celebros, gléowine |

**Reading the numbers:**

- Loss is **not** comparable to the original char-level project (~0.5), nor directly
  across 256 ↔ 512: BPE spreads probability over a larger vocabulary and fewer
  tokens per name, so the absolute cross-entropy differs by construction. Within a
  single vocab size the architecture ranking is meaningful.
- The vocab-512 models are **noticeably larger** — the embedding and output head
  (`vocab × hidden`) dominate at this scale (Gemma: 113k → 171k, +50%). Bigger vocab
  buys shorter sequences and lower loss, but the models are no longer as "tiny".

## Reproduce from scratch

Requires Python 3.13 and [`uv`](https://github.com/astral-sh/uv). Dependencies:
`torch` (CPU), `tokenizers`, `numpy`.

```bash
uv venv                                                   # create .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install tokenizers numpy

python src/scrape_locations.py       # -> data/locations.txt
python src/clean_data.py             # -> data/middle_earth_names.txt
python src/bpe_tokenizer.py          # -> bpe/bpe_256.json, bpe/bpe_512.json
python src/train_all.py              # -> checkpoints/*.pt (8 models, a few minutes)

python src/generate.py qwen3_5 512 --count 20 --temperature 0.8
```

### Generation options

```bash
python src/generate.py <arch> <vocab> [--count N] [--temperature T] [--seed S] [--novel-only]
```

- `arch` ∈ {qwen3, qwen3_5, gemma4, deepseek3}, `vocab` ∈ {256, 512}
- Lower `--temperature` → safer, more familiar names; higher → more inventive.
- `--novel-only` hides names that already exist in the training corpus.
