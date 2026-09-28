"""Always generate free-form replies — no preset help/status cards."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from ..reasoner import Step, Trace
from ..tools import Tool, call_tool

if TYPE_CHECKING:
    from .intent import IntentNet
    from .markov import MarkovGen
    from .runtime import NanoBrain
    from .wordlm import WordLM


def _args(name: str, text: str) -> dict[str, Any]:
    t = text.strip()
    if name == "calc":
        m = re.search(r"([0-9][0-9\.\s\+\-\*\/\%\(\)\^x]*)", t)
        return {"expr": (m.group(1).strip() if m else t)}
    if name == "search":
        q = re.sub(
            r"^(what is|what's|who is|search(?: the)?(?: web)?|look up|explain|define|tell me about|google)\s+",
            "",
            t,
            flags=re.I,
        ).strip("?.! ")
        return {"query": q or t}
    if name == "remember":
        m = re.match(r"remember(?:\s+that)?\s+(.+?)\s+(?:is|=)\s+(.+)$", t, re.I)
        if m:
            return {"key": m.group(1).strip(), "value": m.group(2).strip()}
        return {"key": t, "value": ""}
    if name == "recall":
        m = re.match(r"^(?:recall|remind me(?: of)?|what(?:'s| is) my)\s+(.+)$", t, re.I)
        return {"key": (m.group(1) if m else t).strip("?.! ")}
    if name == "open":
        m = re.match(r"^(?:open|launch|start)\s+(.+)$", t, re.I)
        if not m:
            return {"target": ""}
        return {"target": m.group(1).strip()}
    if name == "shell":
        m = re.match(r"^(?:run|shell|exec)\s+(.+)$", t, re.I)
        return {"cmd": (m.group(1) if m else t).strip()}
    if name == "note":
        m = re.match(r"^(?:note|jot|take a note)\s*[:\-]?\s*(.+)$", t, re.I)
        return {"text": (m.group(1) if m else t).strip()}
    if name == "find":
        m = re.match(r"^(?:find file|find|locate)\s+(.+)$", t, re.I)
        return {"name": (m.group(1) if m else t).strip()}
    return {}


def _fact(intent: str, args: dict, obs: str) -> str:
    obs = (obs or "").strip()
    if not obs:
        return ""
    if intent == "calc" and not obs.startswith("calc error"):
        return f"{args.get('expr','?')} = {obs}"
    if intent == "remember":
        return f"{args.get('key')} is {args.get('value')}"
    if intent == "recall":
        return obs.replace("**", "")
    if intent == "time":
        return str(obs)
    if intent == "sysinfo":
        lines = obs.splitlines()
        bits = [ln for ln in lines if ln.startswith(("OS:", "RAM:", "Host:"))]
        return "; ".join(bits) if bits else lines[0] if lines else obs
    if intent == "search":
        if "unavailable" in obs.lower():
            return ""
        return re.sub(r"\s+", " ", obs.split("\n\n")[0])[:220]
    if intent == "shell":
        return obs[:180]
    if intent == "open":
        return f"opened {args.get('target')}"
    if intent == "note":
        return f"note:{args.get('text')}"
    if intent == "find":
        return obs[:180]
    return obs[:180]


def _bad(s: str) -> bool:
    if not s or len(s.strip()) < 2:
        return True
    if sum(c.isalpha() for c in s) < 2:
        return True
    low = s.lower()
    if any(t in low for t in ("<bot", "<user", "<tool", "<think")):
        return True
    return False


class NanoReasoner:
    """Intent → optional tool → **generate** spoken answer every time."""

    def __init__(
        self,
        tools: dict[str, Tool],
        intent: "IntentNet | None" = None,
        brain: "NanoBrain | None" = None,
        wordlm: "WordLM | None" = None,
        markov: "MarkovGen | None" = None,
    ):
        self.tools = tools
        self.intent = intent
        self.brain = brain
        self.wordlm = wordlm
        self.markov = markov

    def think(self, user_text: str) -> Trace:
        text = (user_text or "").strip()
        trace = Trace(mode="generate")
        tag = []
        if self.intent:
            tag.append(f"intent:{self.intent.param_count():,}")
        if self.markov:
            tag.append(f"markov:{self.markov.param_count():,}")
        if self.wordlm:
            tag.append(f"wordlm:{self.wordlm.param_count():,}")
        if self.brain:
            tag.append("nanolm")
        trace.steps.append(Step("**generating** · " + " ".join(tag)))

        route, conf = "chat", 0.0
        if self.intent is not None:
            route, probs = self.intent.predict(text)
            conf = max(probs)
            from .intent import TOOLS

            top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
            board = ", ".join(f"{TOOLS[i]}={probs[i]:.2f}" for i in top)
            trace.steps.append(Step(f"intent **{route}** {conf:.0%} [{board}]"))
            # hard guards against common confusions
            low = text.lower().strip()
            if route == "open" and not re.match(r"^(open|launch|start)\b", low):
                route = "search" if low.startswith(("tell me", "what", "who", "why", "how", "explain")) else "chat"
                trace.steps.append(Step(f"guard: open→{route}"))
            if route == "recall" and low.startswith("remember"):
                route = "remember"
            if re.match(r"^(?:tell me about|what is|what's|who is|why is|explain|define)\b", low):
                if route in {"open", "chat", "help", "note", "find", "sysinfo"}:
                    # "who are you" is chat identity, not search/sysinfo
                    if re.search(r"\b(you|your name|lito)\b", low):
                        route = "chat"
                        trace.steps.append(Step("guard: identity→chat"))
                    else:
                        route = "search"
                        trace.steps.append(Step("guard: knowledge→search"))
            if route == "sysinfo" and re.search(r"\b(who are you|your name|how are you)\b", low):
                route = "chat"
                trace.steps.append(Step("guard: sysinfo→chat"))
        else:
            route = self._guess(text)

        args: dict[str, Any] = {}
        obs = ""
        if route in self.tools and route not in {"chat", "help"}:
            args = _args(route, text)
            trace.steps.append(Step(f"tool:{route}"))
            try:
                obs = str(call_tool(self.tools, route, args))
            except Exception as exc:
                obs = f"error {exc}"
            trace.steps[-1].observation = obs[:350]

        fact = _fact(route, args, obs)

        # ---- generate free-form answer ----
        answer = ""
        toolish = route in {
            "calc", "remember", "recall", "time", "sysinfo", "shell", "open", "note", "find"
        }

        # Tool-grounded routes: write a spoken line from the fact (still varied)
        if toolish and (fact or obs):
            answer = self._speak_fact(route, args, fact, obs)
            trace.steps.append(Step("generated grounded reply from tool fact"))
        else:
            samples: list[str] = []
            if self.markov is not None:
                for temp in (0.9, 1.05, 0.75):
                    try:
                        samples.append(
                            self.markov.generate(
                                text, fact=fact if route == "search" else "", temperature=temp
                            )
                        )
                    except Exception:
                        pass
                if samples:
                    trace.steps.append(Step(f"generated {len(samples)} candidates"))

            if self.wordlm is not None:
                try:
                    samples.append(
                        self.wordlm.generate(
                            f"{text} [info {fact}]" if fact else text,
                            max_words=36,
                            temperature=0.8,
                        )
                    )
                except Exception:
                    pass

            if self.brain is not None:
                try:
                    from .generate import generate_reply

                    samples.append(
                        generate_reply(
                            self.brain.model,
                            self.brain.tok,
                            text,
                            obs=fact if route == "search" else "",
                            max_new=48,
                            temperature=0.85,
                        )
                    )
                except Exception:
                    pass

            def score(s: str) -> float:
                if _bad(s):
                    return -10.0
                sc = min(len(s), 160) / 40.0
                if fact and any(p in s.lower() for p in fact.lower().split()[:4] if len(p) > 3):
                    sc += 2.5
                if re.search(r"[.!?]$", s.strip()):
                    sc += 0.4
                return sc

            good = [(score(s), s) for s in samples if not _bad(s)]
            if good:
                good.sort(key=lambda x: -x[0])
                answer = good[0][1]
                trace.steps.append(Step(f"chose generated text (score {good[0][0]:.1f})"))

        if _bad(answer):
            answer = self._improvise(text, route, args, fact, obs)
            trace.steps.append(Step("improvised fluent line"))

        trace.answer = answer.strip()
        return trace

    def _speak_fact(self, route: str, args: dict, fact: str, obs: str) -> str:
        R = random.choice
        if route == "calc" and obs and not str(obs).startswith("calc error"):
            expr = args.get("expr", "")
            return R(
                [
                    f"I calculate {expr} = {obs}.",
                    f"Working it out: {expr} equals {obs}.",
                    f"That comes to {obs} ({expr}).",
                    f"Result: {obs}.",
                    f"Got {obs} for {expr}.",
                ]
            )
        if route == "remember":
            k, v = args.get("key", ""), args.get("value", "")
            return R(
                [
                    f"Got it — I will remember that {k} is {v}.",
                    f"Saved. {k} is stored as {v}.",
                    f"Okay, remembering {k} = {v}.",
                    f"Noted in memory: {k} is {v}.",
                ]
            )
        if route == "recall":
            if "nothing" in (obs or "").lower():
                key = args.get("key", "that")
                return R(
                    [
                        f"I do not have anything stored for {key} yet.",
                        f"Nothing in memory under “{key}”.",
                        f"I have not learned {key} so far.",
                    ]
                )
            # obs is like "pet = cat"
            return R(
                [
                    f"From memory: {obs}.",
                    f"I recall {obs}.",
                    f"Looking it up — {obs}.",
                    f"Stored value: {obs}.",
                ]
            )
        if route == "time":
            return R([f"The local time is {obs}.", f"Right now it is {obs}.", f"Clock says {obs}."])
        if route == "sysinfo":
            return R(
                [
                    f"System snapshot: {fact or obs.splitlines()[0] if obs else 'ok'}.",
                    f"Here is what the machine reports: {fact or obs[:160]}.",
                ]
            )
        if route == "shell":
            return f"Command output:\n{obs}"
        if route == "open":
            tgt = args.get("target") or ""
            if not tgt:
                return R(["What should I open?", "Give me a URL or app name to open."])
            return R([f"Opening {tgt}.", f"Launched {tgt}."])
        if route == "note":
            return R([f"Noted: {args.get('text')}.", f"I wrote that down — {args.get('text')}."])
        if route == "find":
            return obs or f"No files matched “{args.get('name')}”."
        return fact or obs or ""

    def _improvise(self, user: str, route: str, args: dict, fact: str, obs: str) -> str:
        """Procedural free-form sentence builder — different each call, not a card."""
        u = user.strip()
        low = u.lower()
        R = random.choice

        if route in {"chat", "help"}:
            if any(low == g or low.startswith(g + " ") for g in ("hi", "hello", "hey", "yo")):
                return R(
                    [
                        f"Hey — what is going on?",
                        f"Hello! I generate my replies as I go; what do you need?",
                        f"Hi. I am here. Ask me something real.",
                        f"Hey there. What should we talk about?",
                        f"Hello. I am listening.",
                    ]
                )
            if "thank" in low:
                return R(["You are welcome.", "Anytime.", "Glad to help.", "Sure thing."])
            if low in {"help", "?", "what can you do"}:
                return R(
                    [
                        "Talk to me like a person. I will figure out tools if needed and write an answer.",
                        "Ask a question, give me math, or tell me to remember something — I respond in sentences.",
                        "No menu required. Just say what you want and I will generate a reply.",
                    ]
                )
            if "who are you" in low or "your name" in low:
                return R(
                    [
                        "I am Lito. I run a tiny neural stack and I write answers instead of reading scripts.",
                        "Lito-Nano — small model, generated replies, tools when the truth matters.",
                    ]
                )
            # Reflective generation from user words
            cleaned = re.sub(r"[^\w\s]", "", u).strip()
            if cleaned:
                return R(
                    [
                        f"On “{cleaned}”: I do not keep a canned speech for that. Tell me if you want an explanation, an opinion, or next steps.",
                        f"Interesting thought — “{cleaned}”. Want me to unpack it, challenge it, or help you act on it?",
                        f"I hear “{cleaned}”. Give me the goal behind that and I will answer more sharply.",
                    ]
                )
            return R(["Say more and I will answer in full.", "I am here — expand a little."])

        if fact:
            return R(
                [
                    f"{fact}.",
                    f"Here is what I have: {fact}.",
                    f"So — {fact}.",
                    f"Putting it together: {fact}.",
                ]
            )
        if obs:
            return obs[:400]
        return R(
            [
                f"I am working with “{u}”. Try another wording and I will generate a clearer answer.",
                f"Not enough signal yet on “{u}”. Add a detail and I will reply in prose.",
            ]
        )

    def _guess(self, text: str) -> str:
        low = text.lower().strip()
        if re.search(r"[0-9]\s*[\+\-\*\/\^]", text) or low.startswith(("calc", "calculate")):
            return "calc"
        if low.startswith("remember"):
            return "remember"
        if low.startswith(("recall", "remind")):
            return "recall"
        if low.startswith(("what is", "who is", "search", "look up", "explain", "define", "tell me")):
            return "search"
        if low.startswith(("time", "date")):
            return "time"
        if "ram" in low or low.startswith("sys"):
            return "sysinfo"
        if low in {"help", "?"}:
            return "help"
        return "chat"
