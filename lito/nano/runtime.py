"""Lito-Nano runtime: neural tool routing via next-token log-probs + generation."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .model import NanoLM, generate
from .tokenizer import Tokenizer


def default_weights_dir() -> Path:
    env = os.environ.get("LITO_NANO_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "weights"


@dataclass
class NanoPlan:
    raw: str
    think: str = ""
    tool: str = ""
    tool_args: str = ""
    answer: str = ""
    route_scores: dict | None = None


# Candidate tool programs the LM scores (agent action space)
_ROUTES = [
    ("chat", "<bot> <think> greet or chat </think> <ans>"),
    ("help", "<bot> <think> list capabilities </think> <ans>"),
    ("calc", "<bot> <think> do math </think> <tool> calc"),
    ("search", "<bot> <think> need facts </think> <tool> search"),
    ("remember", "<bot> <think> store memory </think> <tool> remember"),
    ("recall", "<bot> <think> lookup memory </think> <tool> recall"),
    ("time", "<bot> <think> clock </think> <tool> time"),
    ("sysinfo", "<bot> <think> system status </think> <tool> sysinfo"),
    ("open", "<bot> <think> open target </think> <tool> open"),
    ("shell", "<bot> <think> run command </think> <tool> shell"),
    ("note", "<bot> <think> save note </think> <tool> note"),
    ("find", "<bot> <think> find file </think> <tool> find"),
]


class NanoBrain:
    def __init__(self, model: NanoLM, tok: Tokenizer):
        self.model = model
        self.tok = tok
        self.max_new = int(os.environ.get("LITO_NANO_MAX_NEW", "40"))
        self.temperature = float(os.environ.get("LITO_NANO_TEMP", "0.4"))

    @property
    def info(self) -> str:
        c = self.model.config
        return (
            f"Lito-Nano {self.model.param_count():,} params · "
            f"d={c.n_embd} H={c.n_hidden} K={c.n_ctx} V={c.vocab_size}"
        )

    def mean_nll(self, token_ids: Sequence[int]) -> float:
        """Average negative log-likelihood of teacher-forced sequence (skip first)."""
        if len(token_ids) < 2:
            return 99.0
        total = 0.0
        n = 0
        # evaluate only last L positions for speed
        start = max(1, len(token_ids) - 24)
        for end in range(start, len(token_ids)):
            logits = self.model.logits_at(token_ids, end)
            target = token_ids[end]
            m = max(logits)
            ex = [math.exp(v - m) for v in logits]
            z = sum(ex)
            p = ex[target] / z
            total += -math.log(max(p, 1e-12))
            n += 1
        return total / max(n, 1)

    def route(self, user_text: str) -> tuple[str, dict[str, float]]:
        """Pick tool/intent by which continuation the LM finds most likely."""
        prefix = f"<user> {user_text.strip()} "
        pref_ids = self.tok.encode(prefix, add_bos=True)
        scores: dict[str, float] = {}
        for name, cont in _ROUTES:
            ids = pref_ids + self.tok.encode(cont)
            # score only the continuation tokens
            if len(ids) <= len(pref_ids) + 1:
                scores[name] = 99.0
                continue
            # NLL of continuation given full sequence teacher force
            total = 0.0
            n = 0
            for end in range(len(pref_ids), len(ids)):
                logits = self.model.logits_at(ids, end)
                target = ids[end]
                m = max(logits)
                exps = [math.exp(v - m) for v in logits]
                z = sum(exps)
                p = exps[target] / z
                total += -math.log(max(p, 1e-12))
                n += 1
            scores[name] = total / max(n, 1)
        best = min(scores, key=scores.get)
        return best, scores

    def plan(self, user_text: str, *, memory_hint: str = "") -> NanoPlan:
        user_text = (user_text or "").strip()
        route, scores = self.route(user_text)
        plan = NanoPlan(raw=f"route={route}", tool=route if route not in {"chat", "help"} else "", think=f"neural route → {route}", route_scores=scores)

        # Extract args from user text (deterministic — LM chooses *which* tool)
        plan.tool_args = self._args_for(route, user_text)

        if route == "chat":
            plan.answer = ""  # reasoner fills via local chat
            plan.tool = ""
        if route == "help":
            plan.tool = ""
            plan.think = "neural route → help"
        return plan

    def _args_for(self, route: str, text: str) -> str:
        t = text.strip()
        if route == "calc":
            m = re.search(r"([0-9][0-9\.\s\+\-\*\/\%\(\)\^x]*)", t)
            return m.group(1).strip() if m else t
        if route == "search":
            q = re.sub(
                r"^(what is|what's|who is|search(?: the)?(?: web)?|look up|explain|define|tell me about)\s+",
                "",
                t,
                flags=re.I,
            )
            return q.strip("?.! ") or t
        if route == "remember":
            m = re.match(r"remember(?:\s+that)?\s+(.+?)\s+(?:is|=)\s+(.+)$", t, re.I)
            if m:
                return f"{m.group(1).strip()}={m.group(2).strip()}"
            return t
        if route == "recall":
            m = re.match(r"^(?:recall|remind me(?: of)?|what(?:'s| is) my)\s+(.+)$", t, re.I)
            return (m.group(1) if m else t).strip("?.! ")
        if route == "open":
            m = re.match(r"^(?:open|launch|start)\s+(.+)$", t, re.I)
            return (m.group(1) if m else t).strip()
        if route == "shell":
            m = re.match(r"^(?:run|shell|exec)\s+(.+)$", t, re.I)
            return (m.group(1) if m else t).strip()
        if route == "note":
            m = re.match(r"^(?:note|jot)\s*[:\-]?\s*(.+)$", t, re.I)
            return (m.group(1) if m else t).strip()
        if route == "find":
            m = re.match(r"^(?:find file|find|locate)\s+(.+)$", t, re.I)
            return (m.group(1) if m else t).strip()
        return ""

    def complete_answer(self, user_text: str, think: str, tool: str, obs: str) -> str:
        prompt = (
            f"<user> {user_text} <bot> <think> {think} </think> "
            f"<tool> {tool} </tool> <obs> {obs[:160]} </obs> <ans>"
        )
        ids = self.tok.encode(prompt, add_bos=True)
        out = generate(
            self.model,
            ids,
            max_new=28,
            temperature=0.3,
            top_k=8,
            eos_id=self.tok.eos_id,
        )
        text = self.tok.decode(out[len(ids) :])
        m = re.search(r"(.*?)(?:</ans>|$)", text, re.S)
        ans = re.sub(r"</?ans>", "", (m.group(1) if m else text)).strip()
        return ans

    def generate_chat(self, user_text: str, obs: str = "") -> str:
        from .generate import generate_reply

        return generate_reply(
            self.model,
            self.tok,
            user_text,
            obs=obs,
            max_new=80,
            temperature=0.75,
        )


def load_brain(path: Path | None = None) -> NanoBrain | None:
    root = path or default_weights_dir()
    if root.is_file():
        bin_path = root
        tok_path = root.with_suffix(".tok.json")
    else:
        bin_path = root / "lito-nano.bin"
        tok_path = root / "lito-nano.tok.json"
    if not bin_path.exists() or not tok_path.exists():
        return None
    model = NanoLM.load(bin_path)
    tok = Tokenizer.load(tok_path)
    return NanoBrain(model, tok)


def try_load_brain() -> NanoBrain | None:
    try:
        return load_brain()
    except Exception:
        return None


def try_load_intent():
    try:
        from .intent import IntentNet

        path = default_weights_dir() / "lito-intent.bin"
        if path.exists():
            return IntentNet.load(path)
    except Exception:
        return None
    return None
