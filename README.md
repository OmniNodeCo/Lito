# Lito

**Lightest thinking AI** — a real multi-step agent that plans, uses tools, and answers.  
**Very low RAM** because it does **not** load model weights in-process (stdlib only, no torch/electron).

```
think → act (tools) → observe → answer
RAM: a few MB · zero pip deps · optional external 1B model
```

## Why this rewrite

Older Lito was a large rule catalog. **0.2** replaces that with:

1. **Local reasoner** — multi-step planner (classify → tool calls → synthesize)
2. **Tools** — calc, shell, web search, fetch, memory, files, open…
3. **Visible thinking** — short trace so you see *how* it decided
4. **Optional neural core** — talk to Ollama / llama.cpp over HTTP; weights stay out of this process

## Quick start

```bash
python3 run_lito.py              # UI → http://0.0.0.0:8765
python3 run_lito.py --cli        # terminal REPL (lightest)
python3 run_lito.py -c "what is MQTT"
python3 run_lito.py -c "calculate 2^16 - 1"
```

```bash
pip install -e .
lito --cli
```

### Optional: real LLM (still tiny RAM here)

```bash
# terminal 1 — any OpenAI-compatible server, e.g. Ollama
ollama serve
ollama pull llama3.2:1b

# terminal 2
export LITO_LLM_URL=http://127.0.0.1:11434/v1/chat/completions
export LITO_LLM_MODEL=llama3.2:1b
python3 run_lito.py --cli
```

Hide thought traces: `LITO_SHOW_THOUGHTS=0` or `--hide-thoughts`.  
Force pure local (ignore LLM): `LITO_FORCE_LOCAL=1`.

## What it can do

| You say | What happens |
|--------|----------------|
| `what is photosynthesis` | Plans → web/wiki tools → extractive answer |
| `calculate 17*19` / `2^10+5` | Safe AST calculator |
| `remember wifi is orchard-5G` | Long-term KV memory |
| `recall wifi` | Memory lookup |
| `search web MQTT QoS` | Instant answers + links in chat |
| `open https://example.com` | Opens URL |
| `run echo hello` | Shell (dangerous patterns blocked) |
| `read ~/notes.txt` | Read file / list dir |
| `find file report.pdf` | Filename walk under home |
| `note buy milk` / `list notes` | Scratch notes |
| `sysinfo` / `time` | Local system facts |
| `help` | Tools + examples |

## Architecture (kept tiny)

```
lito/
  agent.py      # public Agent.handle()
  reasoner.py   # LocalReasoner + optional LLMReasoner (ReAct-style)
  tools.py      # all side effects
  memory.py     # ~/.local/share/lito/memory.json
  ui.py / cli.py
```

No embedding DB, no browser engine, no CUDA. Memory is a small JSON file.

## Standalone binary

```bash
python3 scripts/build_exe.py
# or: pyinstaller lito.spec
```

CI builds Linux / macOS / Windows on tag `v*`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT
