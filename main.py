import json
import os
import random
import re
import sqlite3
import sys
from datetime import datetime

import joblib

# ──────────────────────────────────────────────
# PATH HELPER (PyInstaller compatibility)
# ──────────────────────────────────────────────
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ──────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lito_memory.db"
)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            intent TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS learned_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            intent TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# DATABASE OPERATIONS
# ──────────────────────────────────────────────
def log_conversation(user_input, bot_reply, intent=None, confidence=0.0):
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (user_input, bot_reply, intent, confidence) VALUES (?, ?, ?, ?)",
        (user_input, bot_reply, intent, confidence),
    )
    conn.commit()
    conn.close()


def save_user_data(key, value):
    conn = get_db()
    conn.execute(
        """INSERT INTO user_profile (key, value, updated_at)
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
        (key, value),
    )
    conn.commit()
    conn.close()


def get_user_data(key):
    conn = get_db()
    row = conn.execute("SELECT value FROM user_profile WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else None


def save_learned_example(sentence, intent):
    conn = get_db()
    conn.execute(
        "INSERT INTO learned_examples (sentence, intent) VALUES (?, ?)",
        (sentence.lower().strip(), intent),
    )
    conn.commit()
    conn.close()


def get_all_learned_examples():
    conn = get_db()
    rows = conn.execute("SELECT sentence, intent FROM learned_examples").fetchall()
    conn.close()
    return [(r["sentence"], r["intent"]) for r in rows]


def get_stats():
    conn = get_db()
    total_msgs = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
    total_learned = conn.execute("SELECT COUNT(*) as c FROM learned_examples").fetchone()["c"]
    conn.close()
    user_name = get_user_data("name")
    return total_msgs, total_learned, user_name


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
def load_model(filename="lito_model.pkl"):
    path = resource_path(filename)
    if not os.path.exists(path):
        print(f"Error: Model file '{filename}' not found!")
        print("Run 'python train.py' first to create the model.")
        input("Press Enter to exit...")
        sys.exit(1)
    return joblib.load(path)


def load_responses(filename="responses.json"):
    path = resource_path(filename)
    if not os.path.exists(path):
        print(f"Error: '{filename}' not found!")
        input("Press Enter to exit...")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25  # Below this = bot doesn't understand


def classify(model, text):
    """
    Classify user input using the trained model.
    Returns (intent, confidence_score).
    """
    cleaned = text.lower().strip()
    if not cleaned:
        return "fallback", 0.0

    # Get probability scores for all intents
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.classes_

    # Find the highest scoring intent
    best_index = probabilities.argmax()
    best_intent = classes[best_index]
    best_confidence = probabilities[best_index]

    # If confidence is too low, treat as fallback
    if best_confidence < CONFIDENCE_THRESHOLD:
        return "fallback", best_confidence

    return best_intent, best_confidence


def get_reply(intent, responses):
    """Pick a random reply for the given intent."""
    replies = responses.get(intent, responses.get("fallback", ["..."]))
    return random.choice(replies)


# ──────────────────────────────────────────────
# NAME DETECTION
# ──────────────────────────────────────────────
def detect_name(text):
    patterns = [
        r"my name is (\w+)",
        r"call me (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"im called (\w+)",
        r"you can call me (\w+)",
    ]
    cleaned = text.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            name = match.group(1).capitalize()
            skip = {"a", "the", "an", "not", "so", "just", "really",
                    "very", "doing", "feeling", "going", "sorry", "fine"}
            if name.lower() not in skip:
                return name
    return None


# ──────────────────────────────────────────────
# RETRAINING (when user teaches new examples)
# ──────────────────────────────────────────────
def retrain_model(model, output_file="lito_model.pkl"):
    """
    Retrain the model with original data + user-taught examples.
    Takes < 1 second even with thousands of examples.
    """
    from training_data import TRAINING_DATA

    sentences = []
    labels = []

    # Original training data
    for intent, examples in TRAINING_DATA.items():
        for example in examples:
            sentences.append(example.lower())
            labels.append(intent)

    # User-taught examples from the database
    learned = get_all_learned_examples()
    for sentence, intent in learned:
        sentences.append(sentence)
        labels.append(intent)

    # Retrain (the pipeline handles TF-IDF + classifier together)
    model.fit(sentences, labels)

    # Save the updated model
    # When running as .exe, save next to the exe, not inside it
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), output_file
    )
    joblib.dump(model, save_path)
    print(f"  Model retrained on {len(sentences)} examples.")


# ──────────────────────────────────────────────
# TEACHING MODE
# ──────────────────────────────────────────────
def teach_mode(model, responses):
    """Let the user teach the bot a new intent + example."""
    print("\n📚 TEACHING MODE")
    print("  Available intents:", ", ".join(sorted(responses.keys())))
    print()

    sentence = input("  Type an example sentence to teach me: ").strip()
    if not sentence:
        print("  Cancelled.")
        return

    intent = input("  What intent/category is this? (e.g. greeting, joke): ").strip().lower()
    if not intent:
        print("  Cancelled.")
        return

    # Check if we have replies for this intent
    if intent not in responses:
        reply = input(f"  I don't have replies for '{intent}' yet. What should I say? ").strip()
        if reply:
            responses[intent] = [reply]
            # Save updated responses to file
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "responses.json"
            )
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4)

    # Save the example to the database
    save_learned_example(sentence, intent)

    # Retrain the model with the new data
    retrain_model(model)

    print(f"  ✅ Learned! When you say something like '{sentence}', I'll classify it as '{intent}'.")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    init_db()
    model = load_model("lito_model.pkl")
    responses = load_responses("responses.json")

    user_name = get_user_data("name")

    print("=" * 50)
    if user_name:
        print(f"🤖 Welcome back, {user_name}! Lito is online.")
    else:
        print("🤖 Lito is online! Type 'exit' to quit.")
    print("   Powered by a trained Naive Bayes model")
    print("   Type 'help' for info | 'stats' for memory | 'teach' to train me")
    print("=" * 50)

    while True:
        try:
            prompt = f"\n{user_name or 'You'}: "
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLito: Catch you later!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            reply = "Catch you later!"
            print(f"Lito: {reply}")
            log_conversation(user_input, reply, "goodbye", 1.0)
            break

        # Special command: teach
        if user_input.lower() in ["teach", "teach me", "train you", "learn this"]:
            teach_mode(model, responses)
            continue

        # Special command: stats
        if user_input.lower() in ["stats", "statistics", "memory"]:
            total_msgs, total_learned, name = get_stats()
            print(f"\n📊 Lito's Memory Stats:")
            print(f"   Total messages logged:  {total_msgs}")
            print(f"   Examples you taught me: {total_learned}")
            print(f"   Your name:              {name or 'Unknown'}")
            print(f"   Model type:             Naive Bayes + TF-IDF")
            print(f"   Database:               {DB_PATH}")
            continue

        # Check for name
        detected_name = detect_name(user_input)
        if detected_name:
            save_user_data("name", detected_name)
            user_name = detected_name
            reply = f"Nice to meet you, {user_name}! I'll remember that."
            print(f"Lito: {reply}")
            log_conversation(user_input, reply, "name_tell", 1.0)
            continue

        # Classify using the trained model
        intent, confidence = classify(model, user_input)
        reply = get_reply(intent, responses)

        # Show confidence in debug mode (optional, remove for cleaner output)
        # print(f"  [debug: {intent} @ {confidence:.0%}]")

        print(f"Lito: {reply}")
        log_conversation(user_input, reply, intent, confidence)


if __name__ == "__main__":
    main()