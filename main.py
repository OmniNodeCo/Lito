import json
import os
import random
import re
import sys


def resource_path(relative_path):
    """
    Get the absolute path to a resource.
    Works for regular Python execution and PyInstaller's temporary folder.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Regular execution: use current script folder
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def load_brain(filename="data.json"):
    """Loads the JSON brain file safely."""
    file_path = resource_path(filename)

    if not os.path.exists(file_path):
        print(f"Error: Brain file not found at: {file_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        input("Press Enter to exit...")
        sys.exit(1)


def clean_text(text):
    """Lowercases and removes punctuation from input."""
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def get_lito_response(user_text, brain):
    cleaned_input = clean_text(user_text)

    # Search through all intent categories
    for category_name, content in brain.items():
        for keyword in content.get("keywords", []):
            clean_keyword = clean_text(keyword)

            # Word-boundary matching to prevent accidental partial matches
            pattern = rf"\b{re.escape(clean_keyword)}\b"

            if re.search(pattern, cleaned_input):
                replies = content.get("replies", [])
                if replies:
                    return random.choice(replies)

    # Fallback if no matching keywords found
    return "I'm not quite sure how to respond to that."


def main():
    brain = load_brain("data.json")

    print("=" * 45)
    print("🤖 Lito is online! Type 'exit' to quit.")
    print("=" * 45)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLito: Catch you later!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Lito: Catch you later!")
            break

        reply = get_lito_response(user_input, brain)
        print(f"Lito: {reply}")


if __name__ == "__main__":
    main()