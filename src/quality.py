"""
Is a generated reply real English, or model noise?

One shared judge, used in two places:

  * src/brain.py - before neural text is shown to a user, and
  * tools/chat_eval.py - the release gate in CI.

A small model that is undertrained, or that is asked something far
outside its training data, produces text that *looks* like writing from
a distance: word-shaped tokens, list markers, confident punctuation,
but invented words and no sentence. The checks here are the cheap,
decisive ones - above all, how many of the words are real English,
which is measured against the word-frequency list the repo already
ships (160k words).

    ok, why = looks_like_english("Hello! How can I help you today?")
    # -> (True, '')
    ok, why = looks_like_english("Use strangespace: Establs can help ...")
    # -> (False, 'invented words')
"""

import re
from typing import Optional, Tuple

_WORD = re.compile(r"[A-Za-z']+")
# "1. ", "2) ", "- ", "* " at the start of a line
_LIST_LINE = re.compile(r'^\s*(\d+[.)]|[-*\u2022])\s+', re.M)

_DICT = None
_DICT_TRIED = False

# Words the frequency list does not carry but that are perfectly fine
# in a Lito reply (its own name, and the contraction stems that a
# word-level list never contains).
_EXTRA_KNOWN = {'lito', 'gguf', 'ai', 'chatml', 'cpu', 'gpu', 'ok', 'okay',
                'hi', 'hey', 'yes', 'no', 'oh', 'um'}

# fraction of >=3 letter words that must be real English words
MIN_KNOWN_RATIO = 0.88
# only judge vocabulary once there are enough words to judge
MIN_WORDS_FOR_RATIO = 6


def _dictionary():
    """The shared English dictionary, loaded once, offline, lazily."""
    global _DICT, _DICT_TRIED
    if not _DICT_TRIED:
        _DICT_TRIED = True
        try:
            from .dictionary import EnglishDictionary
            _DICT = EnglishDictionary(online=False)
            _DICT.is_word('hello')          # force the data load
        except Exception:                   # pragma: no cover
            try:
                from dictionary import EnglishDictionary  # type: ignore
                _DICT = EnglishDictionary(online=False)
                _DICT.is_word('hello')
            except Exception:
                _DICT = None
    return _DICT


def known_word_ratio(text: str, dictionary=None) -> Tuple[float, int]:
    """(fraction of real English words, how many words were judged)."""
    dictionary = dictionary or _dictionary()
    if dictionary is None:
        return 1.0, 0
    words = [w.lower().strip("'") for w in _WORD.findall(text)]
    words = [w for w in words if len(w) >= 3]
    if not words:
        return 1.0, 0
    known = sum(1 for w in words if _is_known(w, dictionary))
    return known / len(words), len(words)


def _is_known(word: str, dictionary) -> bool:
    if word in _EXTRA_KNOWN or dictionary.is_word(word):
        return True
    if word.endswith('s') and dictionary.is_word(word[:-1]):
        return True
    if "'" in word:                      # don't, I'm, it's, Lito's
        stem = word.split("'")[0]
        if stem and (stem in _EXTRA_KNOWN or dictionary.is_word(stem)):
            return True
    return False


def looks_like_english(text: str, dictionary=None,
                       allow_lists: bool = False) -> Tuple[bool, str]:
    """True when `text` reads like a real answer.

    Returns (ok, reason) - reason is '' when ok, otherwise a short tag
    naming the problem, which is useful in logs and test output.
    """
    text = (text or '').strip()
    words = _WORD.findall(text)
    if len(words) < 2:
        return False, 'too short'

    long_words = [w for w in words if len(w) >= 4]
    if len(long_words) < max(1, 0.15 * len(words)):
        return False, 'no real words'

    unique = len({w.lower() for w in words})
    if len(words) > 8 and unique / len(words) < 0.35:
        return False, 'repetitive'

    content = sum(1 for c in text if c.isalnum())
    if content / max(1, len(text)) < 0.55:
        return False, 'mostly punctuation'

    if not allow_lists and len(_LIST_LINE.findall(text)) >= 2:
        return False, 'list of fragments'

    ratio, judged = known_word_ratio(text, dictionary)
    if judged >= MIN_WORDS_FOR_RATIO and ratio < MIN_KNOWN_RATIO:
        return False, f'invented words ({ratio:.0%} real)'

    return True, ''


def describe(text: str, dictionary=None) -> str:
    """One-line quality summary, for logs."""
    ok, why = looks_like_english(text, dictionary)
    ratio, judged = known_word_ratio(text, dictionary)
    state = 'ok' if ok else f'rejected: {why}'
    return f'{state} | {len(text)} chars | {ratio:.0%} real words ({judged})'
