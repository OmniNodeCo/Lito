# SmartAI (Lito)

An AI built **entirely from scratch** — no pretrained models, no ML frameworks, no
API keys. A from-scratch numpy transformer, a from-scratch NLP pipeline, the full
English dictionary, and live web search.

**Current version: v1.2.0** — see [Versions](#versions).

```
  ____                       _      _    ___
 / ___| _ __ ___   __ _ _ __| |_   / \  |_ _|
 \___ \| '_ ` _ \| _` | '__| __| / _ \  | |
  ___) | | | | | | (_| | |  | |_ / ___ \ | |
 |____/|_| |_| |_|\__,_|_|   \__/_/   \_\___|
```

## What it can do

- **Chat and answer questions** using a transformer language model trained from
  scratch, a built-in knowledge base and BM25 knowledge retrieval
- **Natural conversation** — apologies ("sorry"), acknowledgements ("ok",
  "cool"), praise, "help" and other social cues get real conversational
  replies instead of canned non-answers
- **Understand complex sentences** — multi-part questions, negation,
  coreference ("what is photosynthesis?" → "how does *it* work?"), clause
  splitting, spelling corrections and question-type detection
- **The entire English dictionary** — 370,105 words for validity and
  spellcheck, plus 147,306 WordNet lemmas with definitions, synonyms and
  examples (all offline), enriched online by the Free Dictionary API
- **Search the web** — Wikipedia and DuckDuckGo Instant Answers, no API keys
  needed, with caching and graceful offline behavior
- **Learn new words and terms** — whenever it meets a word or term it
  doesn't know, in a question ("what is minecraft?"), a bare mention
  ("minecraft") or a statement ("i have been playing minecraft"), SmartAI
  searches the web, adds it to its own learned dictionary and remembers it
  offline forever after (each learned entry bumps the AI's knowledge version)

## Quick start

```bash
pip install -r requirements.txt   # just numpy
python main.py                    # chat (trains or downloads a model if none found)
```

Or run without a model — the dictionary, knowledge base and search all work
without any training:

```bash
python main.py --chat-only
```

## Example session

```
🧑 You: Why is the sky blue, and how does photosynthesis work?
🤖 AI: The sky appears blue because of Rayleigh scattering. Sunlight entering the
       atmosphere is scattered by gas molecules, and shorter blue wavelengths
       scatter more than longer red wavelengths. Photosynthesis is the process by
       which plants convert sunlight, carbon dioxide, and water into glucose and
       oxygen. This process is essential for life on Earth.

🧑 You: What is photosynthesis?
🤖 AI: Photosynthesis (noun): synthesis of compounds with the aid of radiant
       energy (especially in plants).

🧑 You: how does it work?
🤖 AI: Photosynthesis is the process by which plants convert sunlight, carbon
       dioxide, and water into glucose and oxygen...

🧑 You: I don't like Mondays, but tell me what gravity is anyway
🤖 AI: Understood - I've noted that. Gravity is a fundamental force that attracts
       objects with mass toward each other...

🧑 You: who won the 2022 world cup?
🤖 AI: 🌐 Wikipedia — 2022 FIFA World Cup (https://en.wikipedia.org/wiki/2022_FIFA_World_Cup):
       The 2022 FIFA World Cup was the 22nd...

🧑 You: How do you spell necesarry?
🤖 AI: (I assumed you meant: 'necesarry' -> 'necessary') 'necesarry' is not in the
       English dictionary. Did you mean: necessary?

🧑 You: what is quantum entanglement?
🤖 AI: Quantum entanglement: Quantum entanglement is the phenomenon whereby the
       quantum states of two or more particles become correlated... Source:
       https://en.wikipedia.org/wiki/Quantum_entanglement. That was new to me, so
       I looked it up and added it to my dictionary - knowledge is now at v1.1.1.

🧑 You: what is quantum entanglement?
🤖 AI: Quantum entanglement (term): Quantum entanglement is the phenomenon whereby
       the quantum states of two or more particles become correlated...
       (answered from the learned dictionary, no web needed)
```

## Chat commands

| Command | Description |
|---------|-------------|
| `/define <word>` | Full dictionary entry: definitions, examples, synonyms |
| `/spell <word>` | Check spelling and get suggestions |
| `/synonyms <word>` | List synonyms |
| `/words <prefix>` | Autocomplete words from the dictionary |
| `/learned` | Show words and terms the AI learned from the web |
| `/search <query>` | Search the web (Wikipedia + DuckDuckGo) |
| `/search on` / `off` | Toggle automatic web search |
| `/generate <prompt>` | Raw neural generation from the trained model |
| `/info` | Model, dictionary, version and search status |
| `/reset`, `/help`, `/quit` | Conversation management |

## CLI options

```
python main.py --train              # train the model locally, then chat
python main.py --download           # download a trained model from GitHub Releases
python main.py --chat-only          # skip the model entirely
python main.py --no-search          # disable web search (everything else stays)
python main.py --no-online-dictionary  # offline-only definitions
```

## How it works

```
user input
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ NLP pipeline (src/nlp.py, from scratch)                  │
│ tokenize → segment → clause-split → lemmatize → negation │
│ → question type → coreference → spelling                 │
└──────────────────────────────────────────────────────────┘
   │ one Clause per part of the sentence
   ▼
┌──────────────────────────────────────────────────────────┐
│ Router (src/brain.py)                                    │
│ 1. small talk & identity                                 │
│ 2. dictionary intents: define / spell / is-a-word /      │
│    synonyms (src/dictionary.py, WordNet offline)         │
│    └─ unknown word? search online, LEARN it, remember it │
│ 3. BM25 retrieval over knowledge base + corpus           │
│    (score + weighted-coverage confidence)                │
│ 4. web search fallback (src/search.py, Wikipedia + DDG)  │
│ 5. neural blending from the trained model                │
└──────────────────────────────────────────────────────────┘
   │
   ▼
combined answer (all clause answers joined)
```

### Complex sentence understanding (`src/nlp.py`)

Every message is parsed into **clauses** — a question like *"Why is the sky
blue, and how does photosynthesis work?"* becomes two independent questions.
The pipeline handles:

- **Negation scope** — "I don't like gravity" does not get answered with
  gravity facts
- **Coreference** — pronouns are resolved against the previous topic, so
  "how does it work?" continues the conversation naturally
- **Question types** — what / why / how / who / when / where / which /
  yes-no, plus definition, spelling, synonym and search intents
- **Lemmatization** — irregular forms (`wolves → wolf`, `went → go`) and
  regular morphology (`running → run`, `studies → study`) using WordNet's
  exception tables
- **Spelling** — unknown words get Damerau-Levenshtein suggestions from a
  160k frequency dictionary, and the AI notes its assumptions

### Dictionary (`src/dictionary.py`)

Offline data in `data/` (see `data/README.md` for sources and licenses):

| File | Contents |
|------|----------|
| `english_words.txt` | 370,105 English words |
| `wordnet_dictionary.jsonl.gz` | 147,306 lemmas, definitions, synonyms, examples (WordNet 3.0, parsed from scratch by `tools/build_dictionary.py`) |
| `wordnet_exceptions.tsv` | 5,952 irregular morphology mappings |
| `word_frequencies.json.gz` | 160,572 word frequencies (spelling + keyword weighting) |

### Learning new words (`src/dictionary.py` + `src/brain.py`)

Whenever the AI meets a word or term it doesn't know — in a definition
question ("what is minecraft?"), a bare mention ("minecraft") or a statement
("i have been playing minecraft") — and it is not in WordNet, the learned
dictionary, or the local knowledge base, it:

1. Searches the web for the word or term (a title-match guard rejects
   results for probable misspellings, so typos fall back to "did you mean?")
2. Saves the summary to `data/learned_dictionary.json`
3. Bumps the knowledge version (e.g. v1.2.0 → v1.2.1) and tells the user

Learned entries are used for every future lookup — offline, instantly, across
restarts. Mentioning a learned term again later ("minecraft is my favorite
game") is answered from that knowledge without re-learning. Single unknown
words are also learned automatically from the Free Dictionary API. See what
the AI knows with `/learned`.

### Versions (`src/version.py`)

SmartAI carries two version numbers:

- **Application version** — the software itself (currently **v1.2.0**). Shown
  at startup, in `/info`, in identity answers and via `python main.py --version`.
- **Knowledge version** — starts at the application version and bumps its
  patch number with every word or term the AI learns (v1.2.0 → v1.2.1 → ...).
  Shown when the AI learns something and in `/info` / `/learned`.

### Web search (`src/search.py`)

Free, key-less APIs over the standard library only:

- **Wikipedia** — search API + REST article summaries
- **DuckDuckGo Instant Answers** — abstracts, definitions, related topics

Results are cached to `data/cache/` with a one-week TTL, requests have short
timeouts, and after a network failure the searcher cools down for 60 seconds
so the chat never stalls when offline. When the web is unreachable the AI
says so honestly instead of inventing an answer.

### Training (`train.py`)

```bash
python train.py --epochs 30 --d_model 128 --n_layers 4
```

Trains the from-scratch transformer (numpy autograd engine in `src/tensor.py`,
layers in `src/layers.py`) on the built-in corpus in `src/dataset.py`, saving
weights and the from-scratch BPE tokenizer to `checkpoints/`.

## Tests

```bash
python tests/test_features.py     # 45 tests, no network required
```

## Releases

Two GitHub Actions workflows handle packaging and publishing:

- **Build** (`.github/workflows/build.yml`) — on every push to main it runs
  the tests, trains a fresh model, packages `best_model.npz` +
  `tokenizer.json`, builds standalone Windows/Linux/macOS executables, and
  keeps the `latest-model` release up to date.
- **Release** (`.github/workflows/release.yml`) — publishes a versioned
  release from the latest green build run. It finds the most recent
  successful build, downloads its model, tokenizer and executables, and
  attaches them to a release tagged with the app version of the commit that
  was built (e.g. `v1.1.0`). Trigger it from the Actions tab
  ("Release SmartAI" -> "Run workflow"), or push a `v*` tag.

`python main.py --download` fetches the model from the latest release, so a
published release works with the existing downloader right away.

## Project layout

```
main.py               chat entry point
train.py              training entry point
src/                  all source (model, NLP, dictionary, search, chat)
data/                 offline dictionary data (see data/README.md)
tools/                dictionary build scripts
tests/                feature tests
build.py, build.yml   PyInstaller packaging config
.github/workflows/    CI: build.yml (train + package), release.yml (publish)
```

## License notes

- WordNet 3.0 data: Princeton University license (see `data/WORDNET_LICENSE`)
- English word list: public domain ([dwyl/english-words](https://github.com/dwyl/english-words))
- Frequency data: MIT ([pyspellchecker](https://github.com/barrust/pyspellchecker))
