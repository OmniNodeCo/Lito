"""
Dice extension for Lito.

Rolls dice: "roll a dice", "roll 2d6", "roll d20", "/roll 3d8".
"""

import random
import re

NAME = 'dice'
VERSION = '1.0.0'
DESCRIPTION = 'Rolls dice: "roll 2d6", "roll a die", "/roll 3d8".'
COMMANDS = {'/roll': 'roll dice, e.g. /roll 2d6 (n dice, s sides)'}

_DICE_RE = re.compile(r'\b(\d*)d(\d+)\b', re.IGNORECASE)
# Only an imperative "roll ..." counts ("roll a dice", "roll 2d6"). This
# keeps the extension from hijacking sentences that merely *mention* dice
# (e.g. after the brain's pronoun resolution rewrites a clause).
_CAN_RE = re.compile(
    r'^(?:hey\s+|please\s+|can you\s+|could you\s+|would you\s+|'
    r'i want to\s+|let\'s\s+|let us\s+)*'
    r'roll\b.*\b(dice|die|d\d+|\d+d\d+)\b'
    r'|^roll\s+(?:a\s+|an\s+|the\s+|us\s+)?(?:dice|die)\b',
    re.IGNORECASE)


def _parse(text: str):
    """Return (count, sides) or None."""
    m = _DICE_RE.search(text)
    if not m:
        return None
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    if not 1 <= count <= 100 or not 2 <= sides <= 1000:
        return None
    return count, sides


def can_handle(text: str) -> bool:
    text = (text or '').strip()
    if not _CAN_RE.search(text):
        return False
    if _DICE_RE.search(text):
        return _parse(text) is not None
    return True  # plain "roll a dice/die"


def handle(text: str, brain) -> str:
    parsed = _parse(text) or (1, 6)
    count, sides = parsed
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    if count == 1:
        return f"🎲 You rolled a d{sides} and got a {total}!"
    roll_text = ' + '.join(str(r) for r in rolls)
    return f"🎲 You rolled {count}d{sides}: {roll_text} = {total}"


def handle_command(cmd: str, args: str, brain) -> str:
    return handle((args or '1d6').strip() or '1d6', brain)
