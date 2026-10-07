# SmartAI Data Files

This directory contains the offline English dictionary data used by
`src/dictionary.py` and the NLP pipeline in `src/nlp.py`.

| File | Contents | Source | License |
|------|----------|--------|---------|
| `english_words.txt` | 370,105 English words (validity checks, autocomplete, lemmatization) | [dwyl/english-words](https://github.com/dwyl/english-words) `words_alpha.txt` | Public domain (Unlicense) |
| `wordnet_dictionary.jsonl.gz` | 147,306 lemmas with definitions, synonyms and example sentences (one JSON line per lemma) | [WordNet 3.0](https://wordnet.princeton.edu/) database, parsed by `tools/build_dictionary.py` | Princeton WordNet license (see `WORDNET_LICENSE`) |
| `wordnet_exceptions.tsv` | 5,952 irregular morphology mappings (inflected form -> base form) | WordNet 3.0 `*.exc` files | Princeton WordNet license |
| `word_frequencies.json.gz` | 160,572 words with corpus frequencies (spelling suggestion ranking, keyword weighting) | [pyspellchecker](https://pyspellchecker.readthedocs.io/) `en.json.gz` | MIT |

`data/cache/` (created at runtime) holds cached web-search and online
dictionary API responses, and `data/learned_dictionary.json` (created at
runtime) stores words and terms the AI looked up online and learned. Neither
is committed to the repository; each learned entry bumps the AI's knowledge
version (see `src/version.py`).

## Regenerating the WordNet files

```bash
# 1. Obtain the WordNet 3.0 database (WNDB format), e.g.:
git clone --depth 1 --filter=blob:none --sparse https://github.com/nltk/nltk_data.git
cd nltk_data && git sparse-checkout set packages/corpora/wordnet
unzip packages/corpora/wordnet.zip -d /tmp/wordnet

# 2. Rebuild the compact dictionary files:
python tools/build_dictionary.py --wordnet-dir /tmp/wordnet/wordnet
```
