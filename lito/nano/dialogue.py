"""Natural-language dialogue corpus for generative training.

Plain conversational turns — the model learns to *write answers*, not
emit tool-script presets.
"""

from __future__ import annotations

import random


def chat_pairs() -> list[tuple[str, str]]:
    """(user, assistant) free-form replies — many paraphrases."""
    pairs: list[tuple[str, str]] = []
    # filled below; extended at end

    greets_u = ["hello", "hi", "hey", "hello there", "hi lito", "hey there", "good morning", "good evening", "howdy", "yo"]
    greets_a = [
        "Hey! Good to see you. What is on your mind?",
        "Hello! I am Lito, a tiny neural agent. Ask me anything.",
        "Hi there. I can chat, do math, search, or remember things for you.",
        "Hey. Ready when you are — questions, calculations, whatever you need.",
        "Hello! How can I help you today?",
        "Hi! Tell me what you need and I will work it out.",
    ]
    for u in greets_u:
        for a in greets_a:
            pairs.append((u, a))

    pairs += [
        ("thanks", "You are welcome. Happy to help anytime."),
        ("thank you", "Glad I could help. What else can I do for you?"),
        ("thx", "Anytime. Fire another question if you like."),
        ("bye", "Goodbye! Come back whenever you need me."),
        ("goodbye", "See you later. Take care."),
        ("how are you", "I am running light and ready. How are you?"),
        ("who are you", "I am Lito-Nano, a custom micro language model that thinks with tools."),
        ("what is your name", "My name is Lito. I am a tiny neural assistant."),
        ("are you an ai", "Yes. I am a small custom neural net paired with tools so I stay accurate."),
        ("what can you do", "I can chat freely, calculate, search the web, remember facts, open links, and run safe shell commands."),
        ("help", "Sure. Ask a question in plain language. I will reason, use tools if needed, and write you an answer."),
        ("help me", "Of course. Tell me the problem and I will work through it step by step."),
    ]

    # open-ended chit chat
    pairs += [
        ("tell me a joke", "Why did the tiny model go to school? To improve its weights and measures."),
        ("tell me something interesting", "A bit of fun: honey never spoils. Jars found in ancient tombs were still edible."),
        ("what do you think about coding", "Coding is structured problem solving. I like helping break tasks into clear steps."),
        ("i am bored", "Want a riddle, a quick fact, a math puzzle, or a topic to explore together?"),
        ("i am sad", "I am sorry you feel down. I am here to listen or distract you with something curious."),
        ("i am happy", "That is wonderful to hear. What made your day good?"),
        ("good night", "Good night. Rest well — I will be here when you return."),
        ("good morning", "Good morning! Hope your day starts smoothly. What are we working on?"),
    ]

    # knowledge-style answers (model learns fluent phrasing; tools still ground facts at runtime)
    knowledge = [
        ("what is mqtt", "MQTT is a lightweight publish and subscribe messaging protocol, popular for IoT devices with limited bandwidth."),
        ("what is photosynthesis", "Photosynthesis is how plants turn light, water, and carbon dioxide into sugar and oxygen."),
        ("what is a walrus", "A walrus is a large marine mammal with tusks and flippers, living in Arctic seas."),
        ("explain gravity", "Gravity is the attraction between masses. On Earth it pulls objects toward the ground."),
        ("define recursion", "Recursion is when a function solves a problem by calling itself on a smaller piece until a base case."),
        ("what is python", "Python is a popular high-level programming language known for clear syntax and a huge ecosystem."),
        ("who is ada lovelace", "Ada Lovelace was a nineteenth-century mathematician often called the first computer programmer."),
        ("what is quantum computing", "Quantum computing uses qubits that can be in superpositions, aiming to solve certain problems faster than classical computers."),
        ("tcp vs udp", "TCP is reliable and ordered; UDP is lighter and faster but does not guarantee delivery."),
        ("why is the sky blue", "The sky looks blue because air scatters shorter blue wavelengths of sunlight more than red ones."),
        ("what is love", "Love is a deep feeling of care and connection people share. Poets write books about it; I can only sketch the idea."),
        ("what is happiness", "Happiness is a sense of wellbeing and joy. It often comes from connection, purpose, and small daily moments."),
        ("what is ai", "Artificial intelligence is software that performs tasks that usually need human-like judgment, learning, or language."),
        ("tell me about the ocean", "Oceans cover most of Earth, drive climate, and hold astonishing biodiversity from plankton to whales."),
        ("what is music", "Music is organized sound in time — rhythm, pitch, and texture that move people emotionally."),
    ]
    pairs.extend(knowledge)
    # paraphrase knowledge questions
    for q, a in list(knowledge):
        if q.startswith("what is "):
            topic = q[8:]
            pairs.append((f"tell me about {topic}", a))
            pairs.append((f"explain {topic}", a))
            pairs.append((f"define {topic}", a))

    # tool-grounded natural answers (obs → fluent sentence)
    grounded = []
    for a in range(0, 15):
        for b in range(0, 12):
            s = a + b
            grounded.append((f"calculate {a}+{b}", str(s), f"{a} plus {b} is {s}."))
            grounded.append((f"what is {a}+{b}", str(s), f"The sum of {a} and {b} equals {s}."))
            if b:
                p = a * b
                grounded.append((f"calculate {a}*{b}", str(p), f"{a} times {b} equals {p}."))
    for a, b, r in ((2, 8, 256), (2, 10, 1024), (2, 16, 65536), (6, 7, 42), (12, 12, 144)):
        grounded.append((f"calculate {a}^{b}" if b > 5 else f"calc {a}*{b}", str(r), f"The result is {r}."))
        grounded.append((f"calculate {a}*{b}", str(r) if a * b == r else str(a * b), f"{a} times {b} is {a * b}."))

    # memory style
    for k, v in [("wifi", "orchard"), ("project", "lito"), ("city", "miami"), ("name", "friend")]:
        grounded.append((f"remember {k} is {v}", f"ok {k}", f"Got it. I will remember that {k} is {v}."))
        grounded.append((f"recall {k}", v, f"You told me {k} is {v}."))
        grounded.append((f"what is my {k}", v, f"Your {k} is {v}."))

    grounded.append(("time", "12:00", "The local time is 12:00."))
    grounded.append(("what time is it", "09:30", "Right now it is 09:30 locally."))
    grounded.append(("sysinfo", "Linux 16GB RAM", "Here is a quick system snapshot: Linux 16GB RAM."))
    grounded.append(("how much ram", "8 GB free", "About 8 GB of RAM is free right now."))
    grounded.append(("note buy milk", "saved", "Okay, I noted that: buy milk."))
    grounded.append(("run echo hello", "hello", "The command printed: hello"))
    grounded.append(("open https://example.com", "opened", "Opening https://example.com in your browser."))
    grounded.append(
        ("search mqtt", "MQTT is a pub/sub protocol", "From what I found: MQTT is a pub/sub protocol used a lot in IoT.")
    )
    grounded.append(
        ("what is photosynthesis", "Plants convert light to energy", "Photosynthesis is how plants convert light into chemical energy.")
    )

    pairs = _extend_plain(pairs)
    samples_plain = list(pairs)
    samples_grounded = grounded
    return samples_plain, samples_grounded


def build_generative_corpus() -> list[str]:
    """Strings in LM training format."""
    plain, grounded = chat_pairs()
    out: list[str] = []
    for u, a in plain:
        out.append(f"<user> {u} <bot> {a}")
        # slight noise paraphrases of user side
        if random.random() < 0.2 and not u.endswith("?"):
            out.append(f"<user> {u}? <bot> {a}")
    for u, obs, a in grounded:
        out.append(f"<user> {u} <bot> <obs> {obs} </obs> {a}")
        out.append(f"<user> {u} <bot> {a}")  # also answer without obs tag
    # thinking-in-prose (not tool DSL)
    thoughts = [
        (
            "what should i learn first in coding",
            "Start with basics: variables, loops, and functions in one language like Python. Build tiny projects early.",
        ),
        (
            "how do i stay productive",
            "Pick one important task, set a short timer, remove distractions, then take a real break. Repeat.",
        ),
        (
            "give me advice",
            "Be kind to future you: write things down, sleep enough, and ship small improvements often.",
        ),
        (
            "i failed today",
            "Failure is data. Notice what broke, adjust one thing, and try a smaller step next.",
        ),
        (
            "motivate me",
            "You do not need the whole path. Just the next brave step. Take it now.",
        ),
    ]
    for u, a in thoughts:
        out.append(f"<user> {u} <bot> {a}")

    random.shuffle(out)
    return out


# Ensure chat_pairs includes advice lines used by genmem
def _extend_plain(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs.extend(
        [
            ("motivate me", "You do not need the whole path. Just the next brave step. Take it now."),
            ("i failed today", "Failure is data. Notice what broke, adjust one thing, and try a smaller step next."),
            ("i am stressed", "Breathe once. Pick the smallest next action. You do not have to solve everything at once."),
            ("tell me a story", "A tiny model walked into a server rack and said: I only need a few kilobytes to care."),
        ]
    )
    return pairs
