"""Free-form response generation — always write an answer, never a preset card."""

from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING, List, Sequence

from . import tensor as T
from .model import NanoLM, generate as _raw_generate

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


def _softmax_temp(logits: Sequence[float], temperature: float, top_k: int) -> List[float]:
    scaled = [l / max(temperature, 1e-6) for l in logits]
    if top_k and top_k < len(scaled):
        top = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
        keep = set(top)
        scaled = [scaled[i] if i in keep else -1e9 for i in range(len(scaled))]
    return T.softmax(scaled)


def sample_id(logits: Sequence[float], temperature: float = 0.7, top_k: int = 25) -> int:
    if temperature <= 0.05:
        return max(range(len(logits)), key=lambda i: logits[i])
    probs = _softmax_temp(logits, temperature, top_k)
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def generate_text(
    model: NanoLM,
    tok: "Tokenizer",
    prompt: str,
    *,
    max_new: int = 64,
    temperature: float = 0.75,
    top_k: int = 30,
    stop_substrings: Sequence[str] = ("<user>", "<bot>", "<obs>", "<tool>", "<think>"),
) -> str:
    """Continue prompt with plain language; strip control tags from the result."""
    ids = tok.encode(prompt, add_bos=True)
    out = list(ids)
    eos = tok.eos_id
    gen_text_parts: list[str] = []

    for _ in range(max_new):
        ctx = out[-model.config.block_size :]
        logits = model.logits_last(ctx)
        # mild ban on immediately repeating last token
        if len(out) >= 1:
            logits = list(logits)
            logits[out[-1]] -= 0.75
        nid = sample_id(logits, temperature=temperature, top_k=top_k)
        out.append(nid)
        if nid == eos:
            break
        piece = tok.decode([nid])
        gen_text_parts.append(piece)
        joined = tok.decode(out[len(ids) :])
        low = joined.lower()
        if any(s in low for s in stop_substrings if s != "<obs>"):
            # cut at stop
            for s in stop_substrings:
                if s in low:
                    joined = joined[: low.index(s)]
                    break
            return _clean_answer(joined)
        # stop on sentence end after enough content
        if len(joined) > 40 and re.search(r"[.!?]\s*$", joined.strip()):
            # keep going a little for longer answers unless double punct
            if len(joined) > 90:
                break
    return _clean_answer(tok.decode(out[len(ids) :]))


def _clean_answer(text: str) -> str:
    t = text.strip()
    # remove control tags leftover
    t = re.sub(r"</?(?:user|bot|obs|tool|think|ans|eos|bos|pad|unk)[^>]*>", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    # drop leading bot/user crumbs
    t = re.sub(r"^(bot|user|assistant)\s*[:\-]?\s*", "", t, flags=re.I)
    # if model emitted tool junk, cut
    if "<" in t and ">" in t:
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t


def build_prompt(user: str, *, obs: str = "", intent: str = "") -> str:
    """Prompt that steers the LM to write a natural answer."""
    user = (user or "").strip()
    if obs:
        obs_s = re.sub(r"\s+", " ", obs.strip())[:220]
        return f"<user> {user} <bot> <obs> {obs_s} </obs> "
    return f"<user> {user} <bot> "


def generate_reply(
    model: NanoLM,
    tok: "Tokenizer",
    user: str,
    *,
    obs: str = "",
    intent: str = "",
    max_new: int = 72,
    temperature: float = 0.7,
) -> str:
    prompt = build_prompt(user, obs=obs, intent=intent)
    # lower temperature when grounding on tool obs for fidelity
    temp = 0.35 if obs else temperature
    text = generate_text(
        model,
        tok,
        prompt,
        max_new=max_new,
        temperature=temp,
        top_k=20 if obs else 35,
    )
    if not text or len(text) < 2:
        # second try hotter / colder
        text = generate_text(
            model,
            tok,
            prompt,
            max_new=max_new,
            temperature=0.9 if not obs else 0.25,
            top_k=40,
        )
    return text
