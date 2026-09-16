# Lito

**Lightweight desktop AI** — open apps, run tasks, take notes. Uses **very little RAM** because it ships with a rule-based brain (no model weights in-process) and **zero third-party dependencies** (Python stdlib only).

```
RAM typically a few MB · no Electron · no torch · no browser bundle
```

## Quick start

```bash
# from this repo
python3 run_lito.py              # web UI on http://0.0.0.0:8765
python3 run_lito.py --cli        # terminal only (lightest)
python3 run_lito.py -c "help"    # one-shot command
python3 -m lito --port 8765
```

Optional install:

```bash
pip install -e .    # provides the `lito` command
lito --cli
```

## What it can do

| You say | Lito does |
|--------|-----------|
| `open firefox` / `launch code` | Starts the app (detached) |
| `open https://example.com` | Opens URL in default browser |
| `open ~/Documents` | Opens a folder |
| `list apps` | Shows launchable apps it found |
| `find file report.pdf` | Filename search under your home |
| `note buy milk` / `show notes` | Local notes (`~/.lito/`) |
| `remember wifi is secret` / `what is wifi` | Key/value memory |
| `calc 22 * 7` | Safe arithmetic |
| `run echo hello` | Shell (dangerous patterns blocked) |
| `search web walrus operator` | Opens DuckDuckGo |
| `set volume 40` | Volume (PulseAudio / macOS) |
| `screenshot` | Saves a PNG if a tool exists |
| `system info` / `how much ram` / `time` | Machine + self stats |
| `help` | Full command list |

## Why it’s low-RAM

1. **No neural net loaded** — intents are regex/rules in a few KB of Python.
2. **Stdlib only** — no PyTorch, no Electron, no Node UI framework.
3. **Tiny local UI** — a few static files served by `http.server`.
4. **Optional smarter chat** — point `LITO_LLM_URL` at Ollama/etc. so **weights stay in another process**; Lito only does HTTP.

```bash
# Example: use a local 1B model via Ollama without loading it into Lito
export LITO_LLM_URL=http://127.0.0.1:11434/v1/chat/completions
export LITO_LLM_MODEL=llama3.2:1b
python3 run_lito.py
```

Check footprint anytime: say **`how much ram`** in the chat.

## Config & data

| Path | Purpose |
|------|---------|
| `~/.lito/config.json` | host, port, safe_shell, optional llm_url |
| `~/.lito/memory.json` | remembered facts |
| `~/.lito/notes.json` | notes |
| `~/.lito/history.jsonl` | chat log (capped) |

Override data dir: `export LITO_DATA=/path/to/dir`.

Skip scanning `.desktop` files: `export LITO_NO_DESKTOP_SCAN=1`.

## Tests

```bash
python3 -m unittest tests.test_brain -v
```

## Platform notes

- **Linux** — `xdg-open`, `.desktop` discovery, `pactl` volume.
- **macOS** — `open -a`, `osascript` volume, `screencapture`.
- **Windows** — basic app map (`explorer`, `notepad`, …); more can be added in `lito/apps.py`.

## License

MIT
