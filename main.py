import json
import os
import random
import re
import sqlite3
import sys
from datetime import datetime


# ──────────────────────────────────────────────
# PATH HELPER (for PyInstaller compatibility)
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
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lito_memory.db")


def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS learned_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE,
            reply TEXT NOT NULL,
            times_used INTEGER DEFAULT 0,
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
def log_conversation(user_input, bot_reply):
    """Save a conversation exchange to the database."""
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO conversations (user_input, bot_reply) VALUES (?, ?)",
        (user_input, bot_reply),
    )
    conn.commit()
    conn.close()


def save_user_data(key, value):
    """Save or update a user profile entry (name, favorites, etc.)."""
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO user_profile (key, value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def get_user_data(key):
    """Retrieve a user profile value by key."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT value FROM user_profile WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["value"] if row else None


def save_learned_response(keyword, reply):
    """Save a user-taught response to the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO learned_responses (keyword, reply)
            VALUES (?, ?)
            ON CONFLICT(keyword) DO UPDATE SET reply = excluded.reply
            """,
            (keyword.lower().strip(), reply),
        )
        conn.commit()
    except Exception:
        pass
    conn.close()


def get_learned_response(user_input):
    """Check if any learned keyword matches the user input."""
    conn = get_db_connection()
    rows = conn.execute("SELECT keyword, reply FROM learned_responses").fetchall()
    conn.close()

    cleaned = clean_text(user_input)
    for row in rows:
        pattern = rf"\b{re.escape(clean_text(row['keyword']))}\b"
        if re.search(pattern, cleaned):
            # Increment usage counter
            conn = get_db_connection()
            conn.execute(
                "UPDATE learned_responses SET times_used = times_used + 1 WHERE keyword = ?",
                (row["keyword"],),
            )
            conn.commit()
            conn.close()
            return row["reply"]
    return None


def get_conversation_stats():
    """Return stats about conversations and memory."""
    conn = get_db_connection()
    total_msgs = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
    total_learned = conn.execute(
        "SELECT COUNT(*) as c FROM learned_responses"
    ).fetchone()["c"]
    user_name = get_user_data("name")
    conn.close()
    return total_msgs, total_learned, user_name


# ──────────────────────────────────────────────
# JSON BRAIN LOADER
# ──────────────────────────────────────────────
def load_brain(filename="data.json"):
    """Load the JSON brain file."""
    file_path = resource_path(filename)
    if not os.path.exists(file_path):
        print(f"Error: '{filename}' not found at {file_path}")
        input("Press Enter to exit...")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        input("Press Enter to exit...")
        sys.exit(1)


# ──────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────
def clean_text(text):
    """Lowercase and remove punctuation."""
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", "", text)


# ──────────────────────────────────────────────
# NAME DETECTION
# ──────────────────────────────────────────────
def detect_name(user_input):
    """Try to extract a name from phrases like 'my name is X'."""
    patterns = [
        r"my name is (\w+)",
        r"call me (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"i'm called (\w+)",
        r"im called (\w+)",
        r"you can call me (\w+)",
    ]
    cleaned = clean_text(user_input)
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            name = match.group(1).capitalize()
            # Filter out common false positives
            if name.lower() not in ["a", "the", "an", "not", "so", "just", "really", "very", "doing", "feeling", "going"]:
                return name
    return None


# ──────────────────────────────────────────────
# RESPONSE ENGINE
# ──────────────────────────────────────────────
def get_lito_response(user_text, brain):
    cleaned_input = clean_text(user_text)

    # 1. Check learned (DB) responses FIRST (user-taught stuff takes priority)
    learned_reply = get_learned_response(user_text)
    if learned_reply:
        return learned_reply

    # 2. Check JSON brain
    for category_name, content in brain.items():
        for keyword in content.get("keywords", []):
            clean_keyword = clean_text(keyword)
            pattern = rf"\b{re.escape(clean_keyword)}\b"
            if re.search(pattern, cleaned_input):
                replies = content.get("replies", [])
                if replies:
                    return random.choice(replies)

    # 3. Fallback
    return "I'm not quite sure how to respond to that. Want to teach me what to say? (type 'teach me')"


# ──────────────────────────────────────────────
# TEACHING MODE
# ──────────────────────────────────────────────
def teach_mode():
    """Interactive mode where the user teaches the bot a new response."""
    print("\n📚 TEACHING MODE")
    keyword = input("  What keyword or phrase should I listen for? ").strip()
    if not keyword:
        print("  Cancelled.")
        return
    reply = input(f"  What should I say when someone says '{keyword}'? ").strip()
    if not reply:
        print("  Cancelled.")
        return
    save_learned_response(keyword, reply)
    print(f"  ✅ Got it! I'll say '{reply}' when I hear '{keyword}'.")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    # Initialize everything
    init_db()
    brain = load_brain("data.json")

    # Check if we know the user
    user_name = get_user_data("name")

    print("=" * 50)
    if user_name:
        print(f"🤖 Welcome back, {user_name}! Lito is online.")
    else:
        print("🤖 Lito is online! Type 'exit' to quit.")
    print("   Type 'help' for commands | 'stats' for memory info")
    print("=" * 50)

    while True:
        try:
            prompt = f"\n{user_name or 'You'}: " if user_name else "\nYou: "
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            reply = "\nLito: Catch you later!"
            print(reply)
            log_conversation("[EXIT]", reply)
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            reply = "Lito: Catch you later!"
            print(reply)
            log_conversation(user_input, reply)
            break

        # Special command: teach mode
        if user_input.lower() in ["teach me", "teach", "learn this"]:
            teach_mode()
            continue

        # Special command: stats
        if user_input.lower() in ["stats", "statistics", "memory"]:
            total_msgs, total_learned, name = get_conversation_stats()
            print(f"\n📊 Lito's Memory Stats:")
            print(f"   Total messages logged: {total_msgs}")
            print(f"   Things you taught me:  {total_learned}")
            print(f"   Your name:             {name or 'Unknown'}")
            print(f"   Database location:     {DB_PATH}")
            continue

        # Check if user is telling us their name
        detected_name = detect_name(user_input)
        if detected_name:
            save_user_data("name", detected_name)
            user_name = detected_name
            reply = f"Nice to meet you, {user_name}! I'll remember that."
            print(f"Lito: {reply}")
            log_conversation(user_input, reply)
            continue

        # Normal response
        reply = get_lito_response(user_input, brain)
        print(f"Lito: {reply}")

        # Log to database
        log_conversation(user_input, reply)


if __name__ == "__main__":
    main()