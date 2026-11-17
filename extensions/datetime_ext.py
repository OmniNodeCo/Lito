"""
Date & time extension for Lito.

Answers "what time is it?", "what's the date?", "what day is it today?"
using the local clock, plus "/time".
"""

import re
from datetime import datetime

NAME = "datetime"
VERSION = "1.0.0"
DESCRIPTION = ("Answers what time is it / what is the date / "
               "what day is it today questions.")
COMMANDS = {"/time": "show the current date and time"}

# Note: the lookaheads keep this robust when the brain's pronoun
# resolution rewrites "it" into the previous topic ("what day is X
# today?"), while still ignoring real questions like "what day is
# christmas?" or "what time is the movie?".
_CAN_RE = re.compile(
    r"\bwhat time is it\b(?![^?]*\bin\b)"
    r"|\bwhat time is\b(?![^?]*\b(?:in|for|the|a|an)\b)"
    r"|\bwhat(?:'s| is)\s+(?:the\s+|current\s+)?time\b(?![^?]*\bin\b)"
    r"|\bwhat(?:'s|\s+is)?\s+(?:the\s+|today'?s\s+|todays\s+)?date\b"
    r"|\bwhat day is (?:it|today)\b"
    r"|\bwhat day is\b(?=[^?]*\btoday\b)"
    r"|\bcurrent (?:time|date)\b"
    r"|\bdate and time\b",
    re.IGNORECASE)


def can_handle(text: str) -> bool:
    return bool(_CAN_RE.search(text or ""))


def _now() -> str:
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def handle(text: str, brain) -> str:
    low = (text or "").lower()
    now = datetime.now()
    if "date" in low or "day" in low:
        return "Today is " + now.strftime("%A, %B %d, %Y") + ". 📅"
    if "time" in low:
        return "It is " + now.strftime("%I:%M %p").lstrip("0") + ". 🕐"
    return "It is " + _now() + "."


def handle_command(cmd: str, args: str, brain) -> str:
    return "It is " + _now() + ". 🕐"
