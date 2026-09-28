"""Generative reply memory — match user → trained answer, light mutation."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .dialogue import chat_pairs


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


_STOP = {
    "a", "an", "the", "to", "of", "in", "on", "for", "with", "from", "as", "by",
    "or", "and", "is", "are", "was", "be", "me", "my", "i", "you", "we", "it",
    "do", "does", "did", "can", "what", "who", "how", "why", "when", "where",
}


def _keys(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP and len(w) > 1}


_SYN = {
    "hello": ["Hello", "Hi", "Hey"],
    "hey": ["Hey", "Hi", "Hello"],
    "hi": ["Hi", "Hey", "Hello"],
    "glad": ["glad", "happy"],
    "welcome": ["welcome"],
    "tiny": ["tiny", "small"],
    "ready": ["ready", "here"],
    "anything": ["anything", "whatever you need"],
}


def _mutate(text: str) -> str:
    words = text.split()
    out = []
    for i, w in enumerate(words):
        bare = re.sub(r"[^\w']", "", w)
        low = bare.lower()
        punct = w[len(bare) :] if len(w) > len(bare) else ""
        if i == 0 and low in _SYN and random.random() < 0.45:
            out.append(random.choice(_SYN[low]) + punct)
        elif low in _SYN and random.random() < 0.2:
            rep = random.choice(_SYN[low])
            if bare[:1].isupper():
                rep = rep[:1].upper() + rep[1:]
            out.append(rep + punct)
        else:
            out.append(w)
    return " ".join(out)


class MarkovGen:
    def __init__(self):
        # list of {user, answer, ukeys, akeys}
        self.replies: list[dict] = []
        self.n_pairs = 0

    def observe(self, answer: str, user: str = "") -> None:
        answer = (answer or "").strip()
        user = (user or "").strip()
        if not answer:
            return
        self.replies.append(
            {
                "user": user,
                "answer": answer,
                "ukeys": sorted(_keys(user)),
                "akeys": sorted(list(_keys(answer))[:8]),
            }
        )
        self.n_pairs += 1

    def train_from_dialogue(self) -> None:
        plain, grounded = chat_pairs()
        for u, a in plain:
            self.observe(a, u)
        for u, obs, a in grounded:
            # keep non-arithmetic grounded answers
            if re.search(r"\b(equals|plus|times)\b", a.lower()) and re.search(r"\d", a):
                continue
            self.observe(a, u)

    def generate(
        self,
        user: str,
        *,
        fact: str = "",
        max_words: int = 42,
        temperature: float = 0.95,
        channel: str | None = None,
    ) -> str:
        u = (user or "").strip()
        uk = _keys(u)
        u_words = _words(u)

        # exact / near-exact user match first
        exact = []
        for row in self.replies:
            tu = row["user"].lower().strip()
            if tu == u.lower() or tu.rstrip("?!") == u.lower().rstrip("?!"):
                exact.append(row["answer"])
        if exact:
            text = _mutate(random.choice(exact))
            return self._weave(text, fact)

        # fuzzy score
        scored: list[tuple[float, str]] = []
        for row in self.replies:
            rk = set(row["ukeys"])
            if not rk and not uk:
                continue
            inter = uk & rk
            if not inter and not (set(u_words) & set(_words(row["user"]))):
                continue
            # Jaccard-ish on keys
            union = uk | rk or {1}
            j = len(inter) / len(union)
            # coverage of user keys
            cov = len(inter) / max(len(uk), 1)
            # prefer shorter distance in length of user strings
            ulen = max(len(_words(row["user"])), 1)
            len_pen = abs(ulen - max(len(u_words), 1)) * 0.03
            score = 2.2 * cov + 1.4 * j - len_pen + random.random() * 0.05 * temperature
            # require at least one content overlap for non-empty uk
            if uk and not inter:
                continue
            scored.append((score, row["answer"]))

        if scored:
            scored.sort(key=lambda x: -x[0])
            # if best is weak, fall through
            if scored[0][0] >= 0.35:
                top = scored[:5]
                weights = [max(0.05, s) for s, _ in top]
                ans = random.choices([a for _, a in top], weights=weights, k=1)[0]
                return self._weave(_mutate(ans), fact)

        # no good match — generative reflective line using user words
        return self._weave(self._reflect(u), fact)

    def _reflect(self, u: str) -> str:
        cleaned = re.sub(r"\s+", " ", u).strip() or "that"
        return random.choice(
            [
                f"Thinking about “{cleaned}” — I do not have a fixed speech for it, so I will answer as I go. What angle matters most?",
                f"On “{cleaned}”: tell me if you want an explanation, advice, or just a second voice.",
                f"I hear “{cleaned}”. Give me one more detail and I will write a fuller answer.",
                f"“{cleaned}” is interesting. Want the short take or a deeper dive?",
            ]
        )

    def _weave(self, text: str, fact: str) -> str:
        fact = (fact or "").strip()
        if not fact:
            return text.strip()
        if re.search(r"\d", fact) and "=" in fact:
            return random.choice(
                [
                    f"I calculate {fact}.",
                    f"Working it out: {fact}.",
                    f"That comes to {fact.split('=')[-1].strip()} ({fact}).",
                    f"Result: {fact}.",
                    f"Got {fact.split('=')[-1].strip()} — {fact}.",
                ]
            )
        if fact.lower() in text.lower():
            return text
        # knowledge fact prefers the fact itself as the answer
        if len(fact) > 20:
            return random.choice(
                [
                    fact if fact.endswith((".", "!", "?")) else fact + ".",
                    f"Here is what I know: {fact.rstrip('.')}.",
                    f"In short — {fact.rstrip('.')}.",
                ]
            )
        return f"{text.rstrip('.')} ({fact})."

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"format": "lito-genmem-v4", "replies": self.replies, "n_pairs": self.n_pairs},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "MarkovGen":
        data = json.loads(path.read_text(encoding="utf-8"))
        m = cls()
        fmt = data.get("format", "")
        if fmt in {"lito-genmem-v4", "lito-genmem-v3"}:
            rows = data.get("replies") or []
            for row in rows:
                if "ukeys" in row:
                    m.replies.append(row)
                else:
                    m.observe(row.get("answer", ""), row.get("user", ""))
            m.n_pairs = int(data.get("n_pairs") or len(m.replies))
            return m
        m.train_from_dialogue()
        return m

    def param_count(self) -> int:
        return sum(len(r.get("answer", "")) for r in self.replies)


def train_and_save(path: Path | None = None) -> MarkovGen:
    path = path or Path(__file__).resolve().parent / "weights" / "lito-markov.json"
    m = MarkovGen()
    m.train_from_dialogue()
    m.save(path)
    print(f"genmem n={len(m.replies)} → {path}", flush=True)
    for q in (
        "hello",
        "what is mqtt",
        "thanks",
        "who are you",
        "i am bored",
        "help",
        "how are you",
        "tell me about the ocean",
        "why is the sky blue",
        "what is love",
        "motivate me",
        "i failed today",
    ):
        print(" ", q, "→", m.generate(q), flush=True)
    print(" ", "math", "→", m.generate("calculate 3+4", fact="3+4 = 7"), flush=True)
    return m
