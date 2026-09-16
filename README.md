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

### Standalone executable

```bash
# local one-file binary (needs pyinstaller once)
python3 scripts/build_exe.py
# → dist/lito-<platform>

# or via spec
pip install pyinstaller
pyinstaller lito.spec
```

CI builds Linux / Windows / macOS binaries on every green `Build` run (artifacts) and publishes them to **GitHub Releases** when you push a `v*` tag (see `.github/workflows/release.yml`).

### Auto-update from GitHub Releases

```bash
lito --check-update
lito --install-update
# or in chat:
#   check update
#   install update
```

On startup Lito quietly probes `latest.json` from the newest release (`LITO_AUTO_UPDATE=0` to disable). Frozen executables can self-replace; source installs download the asset for manual swap / `pip install -U`.
## What it can do

| You say | Lito does |
|--------|-----------|
| `open firefox` / `launch code` | Starts the app (detached) |
| `open https://example.com` | Opens URL in default browser |
| `open ~/Documents` | Opens a folder |
| `list apps` / `show all installed apps` | Full inventory of installed apps (no cap) |
| `list apps firefox` | Filter the inventory |
| `refresh apps` | Rescan .desktop / Applications / Start Menu |
| `find file report.pdf` | Filename search under your home |
| `scan caches` | Maps caches → app/system owner; marks in-use vs unused |
| `clear unused caches` | Deletes **unused/orphaned** user caches (skips running apps) |
| `clear unused caches dry run` | Preview only — no deletes |
| `clear cache for firefox` | Clears one owner's cache if that app isn't running |
| `free up cache space` | Same as clear unused (voice-friendly) |
| `note buy milk` / `show notes` | Local notes (`~/.lito/`) |
| `remember wifi is secret` / `what is wifi` | Key/value memory |
| `calc 22 * 7` | Safe arithmetic |
| `run echo hello` | Shell (dangerous patterns blocked) |
| `search web walrus operator` | Opens DuckDuckGo |
| `set volume 40` | Volume (PulseAudio / macOS) |
| `screenshot` | Saves a PNG if a tool exists |
| `system info` / `how much ram` / `time` | Machine + self stats |
| `check update` / `install update` | GitHub Releases auto-update |
| `help` | Full command list |

### Cache cleaner — how it decides

1. **Discovers** known app caches (browsers, editors, chat, package managers) plus `~/.cache/*` and OS cache dirs.
2. **Owns** each path (Firefox, Chrome, pip, APT, thumbnails, …).
3. **Checks running processes** — if the owner is live, status = `in use` and Lito **will not delete**.
4. **Orphaned** caches (app uninstalled, folder left behind) are safe to clear.
5. **Protected** paths (home root, `.ssh`, `.lito`, …) are never removed.
6. System scopes like `/var/cache/apt` need an explicit `including system` and still skip anything protected.

```bash
python3 run_lito.py -c "scan caches"
python3 run_lito.py -c "clear unused caches dry run"
python3 run_lito.py -c "clear unused caches"
python3 run_lito.py -c "clear cache for pip"
```

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

## CI

GitHub Actions workflow: [`.github/workflows/build.yml`](.github/workflows/build.yml)

- Unit tests on Python 3.9 / 3.11 / 3.12 (Ubuntu) + macOS/Windows smoke
- CLI one-shot smoke (`help`, `scan caches`, dry-run clean)
- `python -m build` sdist/wheel + install check
- `compileall` syntax gate

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest tests.test_cache -v
```

## Platform notes

- **Linux** — `xdg-open`, `.desktop` discovery, `pactl` volume.
- **macOS** — `open -a`, `osascript` volume, `screencapture`.
- **Windows** — basic app map (`explorer`, `notepad`, …); more can be added in `lito/apps.py`.

## License

MIT
