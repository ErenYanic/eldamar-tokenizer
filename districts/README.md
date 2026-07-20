# Turkish district (ilçe) name generator

A small **separate** side task inside this repo, deliberately kept apart from the main [eldamar-tokenizer](../README.md) BPE pipeline. Same four tiny architectures, but a different dataset (Turkish district names) and the original repo's plain **character** tokeniser — no BPE.

```
$ python districts/generate_districts.py gemma4 --count 10 --temperature 0.8 --seed 1
karaisalı  tosya    çukurca  lapseki  mudurnu
mudurnu    çukurca  göynü…   erenler  selkit
```

`karaisalı`, `tosya`, `lapseki`, `mudurnu` and `erenler` are real districts the model has learnt; `selkit` and the rest are invented but plausibly Turkish.

## What it reuses (unchanged)

This task **does not touch** the `src/` BPE pipeline. It reuses the original `single_letter_transformers` repo directly:

- `single_letter_transformers/data/temizle_isimler.py` — the repo's Turkish aware cleaner (correct here: it lower-cases with the proper Turkish `I/İ` rules, splits multi-word names, and de-duplicates).
- each architecture's `tokenizer.py` (`CharTokenizer`), `config.py` and `model.py`.

## Data

Source: the Turkish Wikipedia article [**Türkiye'nin ilçeleri**](https://tr.wikipedia.org/wiki/Türkiye'nin_ilçeleri), a single large sortable table where column 1 is the province (il) and **column 2 is the district (ilçe)**. [`scrape_ilceler.py`](scrape_ilceler.py) fetches it through the MediaWiki API (URL-encoding the Turkish title so it can't get corrupted) and keeps the display text of every column-2 link.

- `ilceler_ham.txt` — **973** raw district names (matches the article's stated count).
- `ilceler_temiz.txt` — **950** cleaned tokens, produced by the repo's `temizle_isimler.py`. Vocabulary: **32** characters (29 Turkish letters + newline + the digits `1`/`9`, which leak in from the one numeric district "19 Mayıs" → `19`; left as-is because the brief was to use the repo cleaner unchanged).

## Results

All four architectures trained for a **uniform 5,000 steps** on CPU (baseline `ln 32 ≈ 3.47`). Because they all share the same character tokeniser, the losses are directly comparable across architectures.

| Architecture               | Params | Steps | Final loss | Sample names                                |
| -------------------------- | ------ | ----- | ---------- | ------------------------------------------- |
| **Qwen3** (dense)          | 19.6k  | 5000  | 0.85       | çınar, karaman, paluova, tikmen             |
| **Qwen3.5** (hybrid)       | 42.1k  | 5000  | 0.57       | aydıncık, havsa, başyayla, büyükçekmece     |
| **Gemma** (sliding-window) | 63.4k  | 5000  | **0.54**   | altındağ, gökçeada, sarıoğlan, arsambitözü  |
| **DeepSeek-V3** (MoE)      | 48.0k  | 5000  | 0.62       | çamlıhemşin, konyaaltı, bigadiç, çatalpınar |

## Run it

```bash
python districts/scrape_ilceler.py                                              # -> ilceler_ham.txt (973)
python single_letter_transformers/data/temizle_isimler.py \
       districts/ilceler_ham.txt districts/ilceler_temiz.txt                    # -> ilceler_temiz.txt (950)

for a in qwen3 qwen3_5 gemma4 deepseek3; do python districts/train_districts.py $a; done
python districts/generate_districts.py gemma4 --count 20 --temperature 0.8
```

`train_districts.py <arch> [--steps N]` and `generate_districts.py <arch> [--count N --temperature T --seed S]`. Keep the step count the same across all four for a fair comparison (the default is 5,000 for every architecture).

## Files

```
districts/
├── scrape_ilceler.py       # fetch + extract column 2 (İlçe) from tr.wikipedia
├── ilceler_ham.txt         # 973 raw district names
├── ilceler_temiz.txt       # 950 cleaned tokens (via the repo's temizle_isimler.py)
├── train_districts.py      # train one architecture (CharTokenizer) → checkpoints/<arch>.pt
├── generate_districts.py   # sample district names from a checkpoint
└── checkpoints/            # <arch>.pt (4 models)
```
