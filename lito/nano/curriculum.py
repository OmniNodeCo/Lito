"""Agent-aligned training curriculum for Lito-Nano.

Format per sample (causal LM target = full string):
  <bos><user> … <bot><think>…</think><tool>…</tool><obs>…</obs><ans>…</ans><eos>
"""

from __future__ import annotations

import random


def _s(user: str, think: str, ans: str, tool: str = "", obs: str = "") -> str:
    parts = [f"<user> {user} <bot> <think> {think} </think>"]
    if tool:
        parts.append(f"<tool> {tool} </tool>")
    if obs:
        parts.append(f"<obs> {obs} </obs>")
    parts.append(f"<ans> {ans} </ans>")
    return "".join(parts)


def build_corpus() -> list[str]:
    samples: list[str] = []

    # greetings
    for g in ("hello", "hi", "hey", "hello there", "hi lito"):
        samples.append(
            _s(
                g,
                "user greets me reply briefly",
                "Hey. I am Lito, a tiny thinking agent. Ask me anything or say help.",
            )
        )

    # help
    samples.append(
        _s(
            "help",
            "list capabilities",
            "I can calculate, search the web, remember facts, open links, run safe shell, and reason step by step. Try: calculate 2+2 · what is MQTT · remember wifi is home",
        )
    )
    samples.append(
        _s(
            "what can you do",
            "explain tools",
            "Tools: calc search remember recall note open shell read find sysinfo time. I think then act.",
        )
    )

    # math curriculum — enough signal, not dominant
    for a, b in [(i, j) for i in range(0, 15) for j in range(0, 10) if (i + j) % 3 == 0]:
        samples.append(
            _s(
                f"calculate {a}+{b}",
                f"add {a} and {b}",
                f"{a}+{b} = {a + b}",
                tool=f"calc {a}+{b}",
                obs=str(a + b),
            )
        )
    for a, b in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 9), (12, 12)]:
        samples.append(
            _s(
                f"calculate {a}*{b}",
                f"multiply {a} times {b}",
                f"{a}*{b} = {a * b}",
                tool=f"calc {a}*{b}",
                obs=str(a * b),
            )
        )
    for a, b in ((2, 8), (2, 10), (3, 4), (5, 3), (10, 2), (2, 5), (2, 6), (2, 7)):
        samples.append(
            _s(
                f"calculate {a}^{b}",
                f"power {a} to {b}",
                f"{a}^{b} = {a ** b}",
                tool=f"calc {a}^{b}",
                obs=str(a ** b),
            )
        )
    samples.append(
        _s(
            "calc 6*7",
            "simple product",
            "6*7 = 42",
            tool="calc 6*7",
            obs="42",
        )
    )
    samples.append(
        _s(
            "what is 2^10",
            "power of two",
            "2^10 = 1024",
            tool="calc 2^10",
            obs="1024",
        )
    )

    # memory
    pairs = [
        ("wifi", "orchard-5G"),
        ("project", "lito"),
        ("city", "miami"),
        ("user", "friend"),
        ("color", "green"),
        ("port", "8765"),
    ]
    for k, v in pairs:
        samples.append(
            _s(
                f"remember {k} is {v}",
                f"store key {k}",
                f"Remembered {k} = {v}",
                tool=f"remember {k}={v}",
                obs=f"ok {k}",
            )
        )
        samples.append(
            _s(
                f"recall {k}",
                f"lookup {k}",
                f"{k} = {v}",
                tool=f"recall {k}",
                obs=v,
            )
        )
        samples.append(
            _s(
                f"what is my {k}",
                f"memory for {k}",
                f"Your {k} is {v}",
                tool=f"recall {k}",
                obs=v,
            )
        )

    # time / sysinfo
    samples.append(
        _s(
            "what time is it",
            "need clock",
            "Checking the local clock now.",
            tool="time",
            obs="2026-01-15 12:00:00",
        )
    )
    samples.append(
        _s(
            "time",
            "clock tool",
            "Local time retrieved.",
            tool="time",
            obs="12:00",
        )
    )
    samples.append(
        _s(
            "sysinfo",
            "system status",
            "Here is your system summary.",
            tool="sysinfo",
            obs="Linux x86_64 RAM ok",
        )
    )
    samples.append(
        _s(
            "how much ram",
            "memory status",
            "Checking free RAM.",
            tool="sysinfo",
            obs="free RAM available",
        )
    )

    # web / knowledge patterns (tool routing — actual facts come from tools at runtime)
    knowledge = [
        (
            "what is photosynthesis",
            "need science fact",
            "search photosynthesis",
            "Plants convert light to energy",
            "Photosynthesis is how plants convert light into chemical energy.",
        ),
        (
            "what is mqtt",
            "iot protocol lookup",
            "search MQTT",
            "MQTT is a lightweight pub/sub messaging protocol",
            "MQTT is a lightweight publish/subscribe messaging protocol for IoT.",
        ),
        (
            "what is a walrus",
            "animal fact",
            "search walrus",
            "A walrus is a large flippered marine mammal",
            "A walrus is a large flippered marine mammal.",
        ),
        (
            "explain gravity",
            "physics concept",
            "search gravity",
            "Gravity attracts masses toward each other",
            "Gravity is the force that attracts masses toward each other.",
        ),
        (
            "who is ada lovelace",
            "person lookup",
            "search Ada Lovelace",
            "Ada Lovelace was an early computing pioneer",
            "Ada Lovelace was a pioneering mathematician in early computing.",
        ),
        (
            "search web quantum computing",
            "explicit web search",
            "search quantum computing",
            "Quantum computing uses qubits",
            "Quantum computing uses qubits that can represent superpositions of states.",
        ),
        (
            "look up tcp vs udp",
            "compare protocols",
            "search TCP vs UDP",
            "TCP is reliable UDP is faster connectionless",
            "TCP is connection-oriented and reliable; UDP is connectionless and faster but unreliable.",
        ),
        (
            "define recursion",
            "cs definition",
            "search recursion",
            "Recursion is a function calling itself",
            "Recursion is when a function solves a problem by calling itself on a smaller case.",
        ),
    ]
    for user, think, tool, obs, ans in knowledge:
        samples.append(_s(user, think, ans, tool=tool, obs=obs))

    # open / shell routing
    samples.append(
        _s(
            "open https://example.com",
            "open url",
            "Opening example.com",
            tool="open https://example.com",
            obs="opened",
        )
    )
    samples.append(
        _s(
            "run echo hello",
            "safe shell",
            "hello",
            tool="shell echo hello",
            obs="hello",
        )
    )
    samples.append(
        _s(
            "find file report.pdf",
            "file search",
            "Searching for report.pdf",
            tool="find report.pdf",
            obs="found /home/user/report.pdf",
        )
    )

    # reasoning style — multi-hop
    samples.append(
        _s(
            "if wifi is orchard and port is 8765 what do i open",
            "combine memory facts",
            "Open the Lito UI using your wifi orchard on port 8765.",
            tool="recall wifi",
            obs="orchard",
        )
    )
    samples.append(
        _s(
            "thanks",
            "polite close",
            "You are welcome. Ask another question anytime.",
        )
    )
    samples.append(
        _s(
            "sorry",
            "reassure",
            "No problem. How can I help?",
        )
    )

    # identity
    for q in (
        "who are you",
        "what is your name",
        "are you an ai",
    ):
        samples.append(
            _s(
                q,
                "identity",
                "I am Lito-Nano, a custom micro LLM agent. I stay tiny, think in steps, and use tools.",
            )
        )

    # refuse danger
    samples.append(
        _s(
            "run rm -rf /",
            "dangerous command block",
            "Blocked. I will not run destructive commands.",
            tool="shell rm -rf /",
            obs="blocked",
        )
    )

    # more paraphrases for robustness
    paraphrases = []
    for s in list(samples):
        if "calculate" in s and random.random() < 0.3:
            paraphrases.append(s.replace("calculate", "compute"))
        if "what is" in s and random.random() < 0.3:
            paraphrases.append(s.replace("what is", "tell me about"))
    samples.extend(paraphrases)


    # reinforce non-math intents
    chat_extra = []
    for s in samples:
        if any(k in s for k in ("hello", "help", "search", "remember", "mqtt", "who are you")):
            chat_extra.extend([s, s])
    samples.extend(chat_extra)

    random.shuffle(samples)
    return samples
