# eldamar-tokenizer

A tiny, end-to-end pipeline that **learns to invent Middle-earth names**. It
cleans a dataset of Tolkien character and place names and trains four tiny
from-scratch LLM architectures on it under three tokenisers — a plain character
baseline plus character-level **BPE** at vocab 256 and 512 — for **12 models in
total** (4 architectures × 3 tokenisers).

```
$ python src/generate.py gemma4 512 --count 8 --temperature 0.8 --seed 1
shield    boromir    markhîr    pengolodh
brandyuis echoriath  arches     running
```

`boromir`, `pengolodh` and `echoriath` are real Tolkien names the model has
memorised; `markhîr`, `brandyuis` and `shield` are inventions (none appear in the
training data); `arches` and `running` are everyday English words the dataset keeps,
split out of multi-word place names. All were produced by models small enough to
train on a CPU in a couple of minutes.

## Credit / attribution

The four model architectures come from
[**malibayram/single_letter_transformers**](https://github.com/malibayram/single_letter_transformers),
a beautiful set of from-scratch tiny-LLM implementations (Qwen3, Qwen3.5, Gemma,
DeepSeek-V3). That repository lives unchanged in
[`single_letter_transformers/`](single_letter_transformers/) as plain content; all
credit for the model code belongs to its author. `eldamar-tokenizer` only swaps
the **data** (Turkish names → Middle-earth names), adds a character-level **BPE**
tokeniser alongside the original character tokeniser, and adds the glue to run the
whole sweep.

## What is different from the original

| | Original (`single_letter_transformers`) | This project |
| --- | --- | --- |
| Task | Generate Turkish first names | Generate Middle-earth names |
| Data | 921 cleaned Turkish names | 2,189 cleaned Tolkien names (characters + places) |
| Tokeniser | Character level (~30 tokens) | Character level **and** character-level **BPE** (256 & 512) |
| Models | 1 per architecture | **3 per architecture** (char + BPE-256 + BPE-512) = 12 |

## Pipeline

```
3 CSVs + scraped locations
        │
        ▼
   clean_data.py ─► data/middle_earth_names.txt   (2,189 names)
        │
        ├─► bpe_tokenizer.py ─► bpe/bpe_{256,512}.json    ┐
        └─► CharTokenizer (built inline at train time)    ├─► train_all.py ─► checkpoints/*.pt (12) ─► generate.py
                                                          ┘
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
├── checkpoints/                   # 12 trained models: {arch}_{char,bpe256,bpe512}.pt
├── src/
│   ├── scrape_locations.py        # 1. fetch place names via the MediaWiki API
│   ├── clean_data.py              # 2. merge + clean all sources into one corpus
│   ├── bpe_tokenizer.py           # 3. train / load the character-level BPE tokeniser
│   ├── train_one.py               # 4. train a single (arch, tokeniser) model
│   ├── train_all.py               #    ... orchestrate all 12
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

The **character baseline** (`char`) uses the original repo's `CharTokenizer`, whose
vocabulary is just the 43 symbols in the corpus (42 letters + newline). It needs no
training and no artifact — it is rebuilt from the names file on each run — and gives
us a reference point to judge what BPE actually buys.

## Results

Each of the four architectures was trained for 5,000 steps on CPU under all three
tokenisers — **12 models**. Every one lands far below its uniform-guessing baseline
(`ln 43 ≈ 3.76`, `ln 256 ≈ 5.55`, `ln 512 ≈ 6.24`).

| Architecture | char — params / loss | BPE-256 — params / loss | BPE-512 — params / loss |
| ------------ | -------------------- | ----------------------- | ----------------------- |
| **Qwen3** (dense)          | 20.0k / 1.38 | 26.8k / 1.47  | 35.0k / 1.25 |
| **Qwen3.5** (hybrid)       | 42.4k / 0.90 | 49.2k / 0.76  | 57.4k / 0.64 |
| **Gemma** (sliding-window) | 65.8k / 0.78 | 113.5k / 0.70 | 170.9k / **0.56** |
| **DeepSeek-V3** (MoE)      | 48.4k / 1.23 | 55.2k / 0.94  | 63.4k / 0.82 |

Sample names (temperature 1.0, straight from training):

| Tokeniser | Examples |
| --------- | -------- |
| char    | balar, héoden, laketown, taur-en-faroth, samdalf, malach |
| BPE-256 | siriondir, alcarnor, elfstan, menelvy, clayhanger, beleth |
| BPE-512 | thorondor, entwash, gléowine, drúadan, harondor, sarum |

**Reading the numbers — the trap to avoid:**

- **Loss is only comparable *down a column*, never *across* one.** Within a tokeniser
  all four architectures share the same vocabulary, token stream and baseline, so the
  losses rank the architectures fairly. Across tokenisers the vocabulary (43 vs 256 vs
  512) and the tokens-per-name differ, so the raw cross-entropy measures different
  things — a lower BPE-512 number does **not** mean it models names "better" than the
  char baseline. (A fair cross-tokeniser metric would be bits-per-character.)
- **Bigger vocab ⇒ bigger model.** The tied embedding/output matrix is `vocab × hidden`,
  so char → 256 → 512 inflates every model (Gemma 66k → 114k → 171k). Seeing that cost
  is part of the point of the sweep.
- **Qualitatively**, the character baseline blends whole words freely (`samdalf` =
  Sam + Gandalf, `laketown`), while BPE leans on learned sub-word chunks and tends to
  assemble names from morpheme-like pieces. Both are fun; neither is strictly best.

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
python src/train_all.py              # -> checkpoints/*.pt (12 models, a few minutes)
#   python src/train_all.py --tokenizer char   # just the 4 character baselines

python src/generate.py qwen3_5 512 --count 20 --temperature 0.8
python src/generate.py qwen3_5 char --count 20 --temperature 0.8
```

### Generation options

```bash
python src/generate.py <arch> <tokenizer> [--count N] [--temperature T] [--seed S] [--novel-only]
```

- `arch` ∈ {qwen3, qwen3_5, gemma4, deepseek3}, `tokenizer` ∈ {char, 256, 512}
- Lower `--temperature` → safer, more familiar names; higher → more inventive.
- `--novel-only` hides names that already exist in the training corpus.
