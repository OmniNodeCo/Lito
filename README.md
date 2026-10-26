# Lito

An AI built **entirely from scratch** — no pretrained models, no ML frameworks, no
API keys. A from-scratch numpy transformer, a from-scratch NLP pipeline, the full
English dictionary, and live web search.

**Current version: v1.2.6** — see [Versions](#versions).

```
  ____                       _      _    ___
 / ___| _ __ ___   __ _ _ __| |_   / \  |_ _|
 \___ \| '_ ` _ \| _` | '__| __| / _ \  | |
  ___) | | | | | | (_| | |  | |_ / ___ \ | |
 |____/|_| |_| |_|\__,_|_|   \__/_/   \_\___|
```

## What it can do

- **Desktop GUI** — **`LitoDesktop`**: a modern **Go** app, Ollama-style — it
  runs the Python AI engine (`server.py`) in the background and serves a
  sleek web UI (dark sidebar, chat bubbles, version-database browser with
  progress downloads, background training with a live log, extension
  toggles). It always launches, even when Python is missing: the UI then
  shows a banner explaining what to install and picks the engine up the
  moment it appears. Pure Go standard library, no external modules.
  Release bundles ship the **frozen `LitoEngine`** (Python + numpy + the
  trained model inside), so the desktop app runs with nothing installed.
  From a source checkout the same app falls back to `python server.py`,
  which also serves the identical UI in your browser with no Go at all.
  (The old tkinter GUI was removed in v1.2.1 — Python is only the AI
  engine)
- **Optional extensions** — small plugins in `extensions/` add instant
  skills (calculator, dice, date & time ship built-in). Toggle them per
  session with `/extensions on|off <name>` or in the desktop app; the
  state persists
- **Chat and answer questions** using a transformer language model trained from
  scratch, a built-in knowledge base and BM25 knowledge retrieval
- **A real GGUF chat model** (v1.2.6) — a from-scratch llama-architecture
  transformer (~5.7M parameters) trained on ~52k instruction pairs and
  exported to `lito-chat.gguf`: a single offline file that runs in
  llama.cpp, LM Studio or Ollama like any open-weight model. Tokenizer,
  trainer and GGUF writer are all implemented in this repo — see
  [The real chat model](#the-real-chat-model-gguf-for-llamacpp--lm-studio--ollama-v126)
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
  ("minecraft") or a statement ("i have been playing minecraft"), Lito
  searches the web, adds it to its own learned dictionary and remembers it
  offline forever after (each learned entry bumps the AI's knowledge version)
- **A more advanced model** (v1.1.0) — rotary position embeddings (RoPE),
  RMSNorm, tied embedding/output weights and repetition-penalized sampling,
  all config-gated so v1.0 checkpoints keep loading unchanged

## Quick start

```bash
pip install -r requirements.txt   # just numpy
python server.py                  # the web UI in your browser
go run ./desktop                  # …or the Go desktop app wrapping it (needs Go)

Desktop users don't need any of this: the LitoDesktop release zips bundle
the Go GUI with the frozen LitoEngine (Python + numpy + model included) —
unzip and double-click. And if the engine is ever missing, the app now
sets itself up: a first-run screen downloads the engine (about 30 MB,
once) straight into the app folder — no installer, no terminal.
```

Or run without a model — the dictionary, knowledge base and search all work
without any training:

```bash
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

🧑 You: what is 15% of 200?
🤖 AI: (15/100)*200 = 30 🧮

🧑 You: roll 2d6
🤖 AI: 🎲 You rolled 2d6: 4 + 3 = 7
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
| `/extensions` | List extensions and their state |
| `/extensions on\|off <name>` | Enable/disable an extension (persisted) |
| `/calc <expr>` | Calculator extension — evaluate a math expression |
| `/roll 2d6` | Dice extension — roll dice |
| `/time` | Date & time extension — show current date and time |
| `/info` | Model, dictionary, version and search status |
| `/reset`, `/help`, `/quit` | Conversation management |

## CLI options

```
python server.py                    # API server + web UI (port 5486)
python server.py --port 8000 --no-search --no-browser
go run ./desktop                    # Go desktop app (launches server.py)
desktop/lito-desktop --backend http://127.0.0.1:5486   # reuse a running backend
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
│ 2. optional extensions (calculator, dice, ...)           │
│ 3. dictionary intents: define / spell / is-a-word /      │
│    synonyms (src/dictionary.py, WordNet offline)         │
│    └─ unknown word? search online, LEARN it, remember it │
│ 4. BM25 retrieval over knowledge base + corpus           │
│    (score + weighted-coverage confidence)                │
│ 5. web search fallback (src/search.py, Wikipedia + DDG)  │
│ 6. neural blending from the trained model                │
└──────────────────────────────────────────────────────────┘
   │
   ▼
combined answer (all clause answers joined)
```

### Desktop GUI

Two ways to run the same app:

**1. Go desktop app — `desktop/`.** Ollama-style: a small Go binary runs
the Python AI engine in the background and serves the modern web UI:

```bash
cd desktop && go build      # pure stdlib, no modules to download
./lito-desktop              # opens the UI in your browser automatically
```

Flags: `-port N`, `-backend URL` (talk to an already-running
`python server.py`), `-python PATH`, `-no-browser`.

The app **always launches**, even when the engine can't be found: the UI
shows a banner explaining what's missing and keeps looking in the
background, so it recovers without a restart. If the engine crashes, it is
restarted automatically. Diagnostics land in `lito-desktop.log` next to
the binary (the Windows build has no console window, and no terminal is
ever shown for the engine either).

Release builds ship as `LitoDesktop-<os>-<arch>.zip` bundles containing
the Go binary **plus `LitoEngine/`** — the AI engine frozen with its own
Python, numpy, `data/` and the trained model. **Nothing to install**:
unzip and double-click. If the engine is missing (or you start with just
the GUI binary), a first-run **Set up Lito** screen downloads
`LitoEngine-<platform>.zip` from the latest release, unpacks it next to
the app and starts it — the setup runs entirely inside the app. From a
source checkout (no frozen engine present) the app runs `python
server.py` instead, with a pre-flight check that the interpreter works
and has numpy.

**2. API server + browser — `server.py`.** Same UI, no Go needed:

```bash
python server.py            # http://127.0.0.1:5486
```

`server.py` is the local API (the "ollama serve" of Lito): `/api/chat`,
`/api/models`, `/api/releases` (version database), `/api/download`,
`/api/extensions`, `/api/train` and friends — all localhost-only, standard
library only.

All of them share one feature set (built on the same `ModelManager` and
`ExtensionManager`):

- **Left sidebar** — the model list (your trained `checkpoints/` plus
  everything downloaded), *Download…* opens the **version database**
  browser listing all GitHub Releases with sizes and dates; ⟳ refreshes,
  🗑 deletes, ⚙ *Train…* opens the training dialog
- **Chat pane** — the brain behind the API, including slash
  commands
- **Extensions & settings** — per-extension toggles and a web-search
  toggle that take effect immediately
- **Training** — pick hyperparameters (or the classic architecture) and
  watch `train.py` stream its output live; the finished model is loaded
  automatically

Model loading, downloads and training all run on background threads, so
the UI never freezes. `ModelManager` (`src/model_manager.py`) is the
version-database layer behind it: local discovery, release listing and
download — usable from scripts too.

### Extensions (`extensions/`)

An extension is a single Python file in `extensions/`:

```python
NAME = "calculator"
VERSION = "1.0.0"
DESCRIPTION = "What shows up in /extensions"
COMMANDS = {"/calc": "evaluate a math expression"}

def can_handle(text: str) -> bool:
    ...   # only claim what you can really answer (be conservative!)

def handle(text: str, brain) -> str:
    ...   # return the reply, or '' to decline

def handle_command(cmd: str, args: str, brain) -> str:
    ...   # optional: answer one of your COMMANDS
```

Lito asks every enabled extension in turn after small talk and before the
dictionary; the first non-empty reply wins. Enabled/disabled state is
persisted in `data/extensions.json`. Built-ins: **calculator** ("what is
2 + 2 * 10", "15% of 200"), **dice** ("roll 2d6") and **datetime** ("what
time is it?").

### Advanced model architecture (v1.1.0)

`TransformerLM` gained three config-gated options (stored inside every
checkpoint, so **old v1.0 checkpoints load unchanged**):

| Option | Classic (v1.0) | Advanced (default) |
|--------|----------------|--------------------|
| `pos_encoding` | learned position embeddings | **RoPE** (rotary, per-head, extrapolates past training length) |
| `norm` | LayerNorm | **RMSNorm** (no mean/bias, faster) |
| `tie_weights` | untied output projection | **tied** with the token embedding |

Plus a CTRL-style **repetition penalty** in `generate()` (the brain uses
1.1) so long answers don't loop. All backward passes — attention with
RoPE rotation, RMSNorm, tied projection — are hand-written and verified
against numerical gradients in the test suite. Train the old layout with
`python train.py --classic`.

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
3. Bumps the knowledge version (e.g. v1.0.3 → v1.0.4) and tells the user

Learned entries are used for every future lookup — offline, instantly, across
restarts. Mentioning a learned term again later ("minecraft is my favorite
game") is answered from that knowledge without re-learning. Single unknown
words are also learned automatically from the Free Dictionary API. See what
the AI knows with `/learned`.

### Versions (`src/version.py`)

Lito carries two version numbers:

- **Application version** — the software itself (currently **v1.2.6**). Shown
  at startup, in `/info` and in identity answers.
- **Knowledge version** — starts at the application version and bumps its
  patch number with every word or term the AI learns (v1.2.6 → v1.2.7 → ...).
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
python train.py --epochs 30 --d_model 128 --n_layers 4   # advanced arch (default)
python train.py --classic ...                            # v1.0 architecture
```

Trains the from-scratch transformer (numpy autograd engine in `src/tensor.py`,
layers in `src/layers.py`) on the built-in corpus in `src/dataset.py`, saving
weights and the from-scratch BPE tokenizer to `checkpoints/`. The default
architecture is RoPE + RMSNorm + tied weights; `--classic` trains the v1.0
layout.

### The real chat model: GGUF for llama.cpp / LM Studio / Ollama (v1.2.6)

Next to the tiny built-in engine, Lito ships a *real* from-scratch chat
model — a llama-architecture transformer (RMSNorm, RoPE, SwiGLU, tied
embeddings, ~5.7M parameters) trained on ~52,000 instruction/response
pairs (Stanford Alpaca, formatted as ChatML) plus public-domain prose,
then exported to **GGUF** so it runs fully offline in anything that
speaks llama.cpp: `llama-cli`, LM Studio, Ollama, etc. No pretrained
weights anywhere — the tokenizer (`src/bpe.py`, a GPT-2-style byte-level
BPE), the trainer (`train_big.py`) and the GGUF writer (`src/gguf.py`)
are all implemented in this repo, from scratch.

```bash
# 1. train (downloads the corpus itself; ~90 min on 2 CPU cores)
pip install torch                     # plus numpy, already required
python train_big.py                   # -> checkpoints-big/checkpoint.pt
python train_big.py --resume --lr 1.5e-3   # optional: keep training
# 2. export to GGUF
python export_gguf.py                 # -> lito-chat.gguf
# 3. chat with it, offline (small models like a repeat penalty)
llama-cli -m lito-chat.gguf --temp 0.8 --repeat-penalty 1.1
#    ...or drop the file into LM Studio / Ollama
```

The exported file is self-describing: llama architecture metadata, the
full BPE vocabulary and merges, ChatML chat template, and the
`<|im_start|>` / `<|im_end|>` control tokens — a front-end that
understands GGUF chat templates needs zero configuration. Every release
carries the trained `lito-chat.gguf` as an asset (see Releases).

Honest expectations: at 5.7M parameters (vs billions for ChatGPT-class
models) trained on ~5M tokens, it chats in simple, mostly-coherent
English, follows the ChatML format, answers common instruction-style
prompts and knows its corpus — it is a real language model, not a
retrieval trick, but it is small. The whole pipeline scales: rerun
`train_big.py` with a bigger `--d_model`, more `--steps` and a larger
corpus on a GPU box and export again — nothing else changes.

**Corpus licensing:** the instruction data is Stanford Alpaca
(`tatsu-lab/stanford_alpaca`), which is intended for research purposes
(see its `DATA_LICENSE`); the prose (tiny shakespeare) is public domain.
`data/bigcorpus/SOURCES.md` records this where the corpus is fetched.

## Tests

```bash
python tests/test_features.py       # 69 tests, no network required
python tests/test_gguf_pipeline.py  # 11 tests: BPE, GGUF, exporter
```

The first suite covers the dictionary, NLP pipeline, search parsing,
brain routing, the extension system (including numerical-gradient checks
of the new RoPE / RMSNorm / tied-weight backward passes), the model
manager, the API server, the desktop app sources, the frozen-engine
training shim and the build configuration. The second covers the GGUF
pipeline: GPT-2 byte-map correctness, encode/decode round-trips,
tokenizer save/load, GGUF v3 file layout (header, metadata, tensor
order, 32-byte alignment, f16/f32 values) and — when `torch` is
installed — the trainer and the full checkpoint → GGUF export.

## Releases

Two GitHub Actions workflows handle packaging and publishing:

- **Build** (`.github/workflows/build.yml`) — on every push to main it runs
  the tests, trains a fresh model, packages `best_model.npz` +
  `tokenizer.json`, then builds per platform (Windows, Linux, macOS Apple
  silicon): the frozen **`LitoEngine`** (Python + numpy + data + model
  inside) and the **Go desktop app**, assembling the self-contained
  `LitoDesktop-*.zip` bundles plus the engine-only zips the in-app setup
  downloads. It also keeps the `latest-model` release up to date.
- **Release** (`.github/workflows/release.yml`) — publishes a versioned
  release from a green build run of the exact commit being released,
  waiting for a concurrent build when needed. Trigger it from the Actions
  tab ("Release Lito" -> "Run workflow"), or push a `v*` tag.

The desktop app's *Download…* button lists every published version (the
"version database") and downloads the model into `models/<tag>/`, and its
first-run setup screen fetches the engine itself — the release is what
makes the app fully self-installing.

## Project layout

```
server.py             local API server + web UI backend (Ollama-style)
desktop/              Go desktop app: main.go, go.mod, ui/ (web UI)
train.py              training entry point (small in-app model)
train_big.py          from-scratch llama-arch trainer -> checkpoint
export_gguf.py        checkpoint -> lito-chat.gguf (llama.cpp & friends)
src/                  all source (model, NLP, dictionary, search, chat,
                      extensions loader, model manager / version database,
                      bpe.py + gguf.py for the GGUF pipeline)
extensions/           optional skill plugins (calculator, dice, datetime)
models/               models downloaded from the version database (created)
data/                 offline dictionary data (see data/README.md); the
                      GGUF training corpus is fetched into data/bigcorpus/
tools/                dictionary build scripts
tests/                feature tests (test_features.py, test_gguf_pipeline.py)
build.py, build.yml   PyInstaller packaging config
.github/workflows/    CI: build.yml (train + package + Go desktop), release.yml (publish)
```

## License notes

- WordNet 3.0 data: Princeton University license (see `data/WORDNET_LICENSE`)
- English word list: public domain ([dwyl/english-words](https://github.com/dwyl/english-words))
- Frequency data: MIT ([pyspellchecker](https://github.com/barrust/pyspellchecker))
er](https://github.com/barrust/pyspellchecker))
