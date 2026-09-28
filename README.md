# Lito

**Custom micro-LLM agent** — a real neural brain that stays tiny.

```
Lito-Nano  ≈ 60k weights  ·  pure Python  ·  no torch/numpy  ·  a few MB RAM
```

Not a wrapper around someone else's giant model. **Lito-Nano** is trained
*for this agent*: intent routing + generative polish + tools.

## Neural stack

| Piece | Role | Size |
|-------|------|------|
| **IntentNet** | Supervised MLP — picks tool/intent | ~13k weights |
| **NanoLM** | Tiny generative LM — chat / polish | ~47k weights |
| **Tools** | calc, search, memory, shell, files… | code only |
| **LocalReasoner** | Deterministic fallback | code only |
| Optional `LITO_LLM_URL` | External Ollama / llama.cpp | out-of-process |

IntentNet hits **~99%** tool routing on its curriculum. Tools supply
**ground truth** (math, search, memory) so the tiny net never has to
memorize the world.

## Quick start

```bash
python3 run_lito.py              # UI http://0.0.0.0:8765
python3 run_lito.py --cli
python3 run_lito.py -c "calculate 6*7"
python3 run_lito.py -c "what is MQTT"
python3 run_lito.py -c "remember wifi is home"
```

### Retrain the custom brain

```bash
# intent router (seconds)
python3 -c "from pathlib import Path; from lito.nano.intent import train_intent; train_intent(out=Path('lito/nano/weights/lito-intent.bin'))"

# generative micro-LM (minutes, optional polish)
python3 -m lito.nano.train --steps 900 --dim 32 --hidden 64
```

Weights live in `lito/nano/weights/` and ship with the repo.

### Optional external LLM

```bash
export LITO_LLM_URL=http://127.0.0.1:11434/v1/chat/completions
export LITO_LLM_MODEL=llama3.2:1b
export LITO_PREFER_EXTERNAL=1
```

Force non-neural path: `LITO_FORCE_LOCAL=1`.

## Examples

| You | Lito-Nano |
|-----|-----------|
| `hello` | neural chat |
| `calculate 2^10` | intent→calc tool → **1024** |
| `what is photosynthesis` | intent→search → grounded answer |
| `remember project is lito` | intent→memory write |
| `recall project` | intent→memory read |
| `help` | branded capabilities |

Every reply can show a **thinking** trace: intent probs + tool calls.

## Layout

```
lito/
  agent.py           # Agent.handle()
  nano/
    intent.py        # IntentNet (the sharp router)
    model.py         # NanoLM (micro generator)
    train.py         # LM trainer
    curriculum.py    # agent-aligned data
    reason.py        # wires neural + tools
    weights/         # *.bin shipped weights
  tools.py reasoner.py memory.py ui.py cli.py
```

## Why this is “tiniest but smart”

1. **Custom** — trained on *agent* traces, not generic web text  
2. **Tiny** — tens of thousands of weights, not billions  
3. **Honest** — tools do math/search; the net decides *what* to do  
4. **Stdlib** — zero runtime pip deps; runs anywhere Python does  

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT
