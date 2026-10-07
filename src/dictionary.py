"""
The English Dictionary - offline dictionary backed by WordNet 3.0 and a
370,000-word English word list, with optional online enrichment from the
Free Dictionary API (https://dictionaryapi.dev).

Everything is loaded lazily so that chat startup is instant; the word list
loads on first use, WordNet definitions on first definition lookup, etc.
"""

import gzip
import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Set, Tuple

from .version import __version__, bump_patch

POS_NAMES = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb',
             's': 'adjective', 'x': 'term'}

_POS_RANK = {'n': 0, 'v': 1, 'a': 2, 'r': 3}


class Definition:
    """A single sense of a word."""

    __slots__ = ('word', 'pos', 'text', 'example', 'synonyms', 'source')

    def __init__(self, word: str, pos: str, text: str,
                 example: str = '', synonyms: Optional[List[str]] = None,
                 source: str = 'wordnet'):
        self.word = word
        self.pos = pos
        self.text = text
        self.example = example
        self.synonyms = synonyms or []
        self.source = source

    @property
    def pos_name(self) -> str:
        return POS_NAMES.get(self.pos, self.pos)

    def __repr__(self):
        return f"Definition({self.word!r}/{self.pos}: {self.text[:40]!r})"


class WordEntry:
    """All senses found for a word."""

    __slots__ = ('word', 'definitions', 'found', 'suggestions', 'freshly_learned')

    def __init__(self, word: str, definitions: Optional[List[Definition]] = None,
                 found: bool = False, suggestions: Optional[List[str]] = None):
        self.word = word
        self.definitions = definitions or []
        self.found = found
        self.suggestions = suggestions or []
        # True when this lookup had to learn the word online just now
        self.freshly_learned = False

    @property
    def parts_of_speech(self) -> List[str]:
        seen = []
        for d in self.definitions:
            if d.pos_name not in seen:
                seen.append(d.pos_name)
        return seen

    def __repr__(self):
        return f"WordEntry({self.word!r}, {len(self.definitions)} senses, found={self.found})"


class EnglishDictionary:
    """
    The entire English dictionary, offline-first, and able to learn.

    - Word validity, autocomplete and spelling suggestions come from a
      370k-word list plus 160k frequency counts.
    - Definitions, synonyms and examples come from WordNet 3.0.
    - When online, entries are enriched with the Free Dictionary API.
    - Words and terms the AI doesn't know are looked up online and saved to
      a persistent learned dictionary (data/learned_dictionary.json), so
      they are known offline forever after. Each learned entry bumps the
      knowledge version.
    """

    WORDS_FILE = 'english_words.txt'
    WORDNET_FILE = 'wordnet_dictionary.jsonl.gz'
    EXCEPTIONS_FILE = 'wordnet_exceptions.tsv'
    FREQ_FILE = 'word_frequencies.json.gz'
    LEARNED_FILE = 'learned_dictionary.json'
    FREE_DICT_API = 'https://api.dictionaryapi.dev/api/v2/entries/en/{}'

    def __init__(self, data_dir: Optional[str] = None, online: bool = True,
                 timeout: float = 4.0, learn_file: Optional[str] = None,
                 autolearn: bool = True):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.data_dir = data_dir
        self.online = online
        self.timeout = timeout
        self.autolearn = autolearn
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        self._learn_path = learn_file or os.path.join(self.data_dir, self.LEARNED_FILE)

        self._words: Optional[Set[str]] = None
        self._entries: Optional[Dict[str, str]] = None     # lemma -> raw JSON string
        self._pos_index: Optional[Dict[str, str]] = None   # lemma -> 'nvar' letters
        self._freq: Optional[Dict[str, int]] = None
        self._exceptions: Optional[Dict[str, List[Tuple[str, str]]]] = None
        self._api_cache: Dict[str, dict] = {}
        self._learned: Optional[dict] = None               # learned dictionary store
        self._load_failed = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    def _ensure_words(self) -> Set[str]:
        if self._words is None and not self._load_failed:
            path = self._path(self.WORDS_FILE)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self._words = {line.strip() for line in f if line.strip()}
            else:
                self._words = set()
        return self._words or set()

    def _ensure_freq(self) -> Dict[str, int]:
        if self._freq is None:
            path = self._path(self.FREQ_FILE)
            if os.path.exists(path):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    self._freq = {k: int(v) for k, v in json.load(f).items()}
            else:
                self._freq = {}
        return self._freq

    def _ensure_entries(self) -> Dict[str, str]:
        if self._entries is None:
            path = self._path(self.WORDNET_FILE)
            entries: Dict[str, str] = {}
            pos_index: Dict[str, str] = {}
            if os.path.exists(path):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        lemma = record.get('w', '')
                        entries[lemma] = line.rstrip('\n')
                        pos_index[lemma] = ''.join(sorted(record.get('p', {}).keys(),
                                                          key=lambda p: _POS_RANK.get(p, 9)))
            self._entries = entries
            self._pos_index = pos_index
        return self._entries

    def _ensure_exceptions(self) -> Dict[str, List[Tuple[str, str]]]:
        if self._exceptions is None:
            path = self._path(self.EXCEPTIONS_FILE)
            exceptions: Dict[str, List[Tuple[str, str]]] = {}
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.rstrip('\n').split('\t')
                        if len(parts) == 3:
                            exceptions.setdefault(parts[1], []).append((parts[0], parts[2]))
            self._exceptions = exceptions
        return self._exceptions

    # ------------------------------------------------------------------
    # Word validity / morphology
    # ------------------------------------------------------------------

    def is_word(self, word: str) -> bool:
        """True if the word (lowercased) is a valid English word."""
        word = word.strip().lower()
        if not word:
            return False
        words = self._ensure_words()
        return word in words or word in self._ensure_entries()

    def pos_of(self, word: str) -> str:
        """Possible parts of speech for a base word, e.g. 'nv'."""
        self._ensure_entries()
        return self._pos_index.get(word.strip().lower(), '')

    def irregular_bases(self, word: str) -> List[Tuple[str, str]]:
        """Irregular base forms for an inflected word: [(pos, base), ...]."""
        return self._ensure_exceptions().get(word.strip().lower(), [])

    def frequency(self, word: str) -> int:
        return self._ensure_freq().get(word.strip().lower(), 0)

    def is_common(self, word: str, threshold: int = 100_000) -> bool:
        return self.frequency(word) >= threshold

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def lookup(self, word: str, use_online: Optional[bool] = None) -> WordEntry:
        """
        Look a word up in the dictionary. Returns all senses found, ordered
        nouns -> verbs -> adjectives -> adverbs (most frequent sense first).

        Lookup order: WordNet -> learned dictionary (words the AI previously
        looked up online) -> Free Dictionary API. When the word is unknown
        locally and the API defines it, the result is saved to the learned
        dictionary (so it is known offline from now on) and the entry is
        flagged `freshly_learned`.
        """
        use_online = self.online if use_online is None else use_online
        raw = word.strip()
        word_l = raw.lower()
        entry = WordEntry(raw)

        # 1. WordNet (offline)
        self._ensure_entries()
        record = None
        for candidate in (word_l, word_l.replace(' ', '_')):
            if candidate in self._entries:
                try:
                    record = json.loads(self._entries[candidate])
                except json.JSONDecodeError:
                    record = None
                break

        if record:
            entry.found = True
            for pos in sorted(record.get('p', {}).keys(), key=lambda p: _POS_RANK.get(p, 9)):
                sense = record['p'][pos]
                for i, text in enumerate(sense.get('d', [])):
                    example = sense.get('e', [''])[i] if i < len(sense.get('e', [])) else ''
                    entry.definitions.append(
                        Definition(raw, pos, text, example,
                                   sense.get('s', [])[:6], source='wordnet'))
        elif word_l and self.is_word(word_l):
            # Valid word but not a WordNet lemma (inflection or rare word)
            entry.found = True

        # 2. Learned dictionary - words/terms the AI looked up before
        if not entry.definitions:
            for candidate in (word_l, word_l.replace(' ', '_')):
                learned = self._learned_entry(candidate)
                if learned:
                    entry.found = True
                    for sense in learned.get('definitions', []):
                        entry.definitions.append(Definition(
                            raw, sense.get('pos', 'x'), sense.get('text', ''),
                            sense.get('example', ''), sense.get('synonyms', []),
                            source='learned'))
                    break

        # 3. Online Free Dictionary API (enrichment, and learning for
        #    words that are unknown locally)
        had_local_definitions = bool(entry.definitions)
        if use_online and word_l:
            extras = self._fetch_online_definitions(word_l)
            if extras:
                entry.found = True
                for pos, text, example, synonyms in extras:
                    entry.definitions.append(Definition(
                        raw, pos, text, example, synonyms, source='online'))
                if not had_local_definitions and self.autolearn:
                    self.learn(word_l, extras, source='free-dictionary-api')
                    entry.freshly_learned = True

        if not entry.found:
            entry.suggestions = self.suggest(word_l)

        return entry

    def define(self, word: str, use_online: Optional[bool] = None) -> List[Definition]:
        return self.lookup(word, use_online=use_online).definitions

    def synonyms(self, word: str) -> List[str]:
        result: List[str] = []
        for definition in self.lookup(word, use_online=False).definitions:
            for syn in definition.synonyms:
                normalized = syn.replace('_', ' ')
                if normalized not in result and normalized != word.lower():
                    result.append(normalized)
        return result

    def _fetch_online_definitions(self, word: str) -> List[Tuple[str, str, str, List[str]]]:
        """Fetch extra definitions from the Free Dictionary API. Returns [] offline."""
        if word in self._api_cache:
            cached = self._api_cache[word]
        else:
            url = self.FREE_DICT_API.format(urllib.parse.quote(word))
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'SmartAI/1.0'})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    json.JSONDecodeError, ValueError, OSError):
                return []
            if not isinstance(data, list):
                return []
            cached = data
            self._api_cache[word] = cached

        results = []
        try:
            for meaning in cached:
                for pos, senses in (meaning.get('meanings') or {}).items():
                    pos_code = {'noun': 'n', 'verb': 'v', 'adjective': 'a',
                                'adverb': 'r'}.get(pos, 'x')
                    for sense in senses[:2]:
                        definition = (sense.get('definition') or '').strip()
                        if not definition:
                            continue
                        example = (sense.get('example') or '').strip()
                        synonyms = (sense.get('synonyms') or [])[:6]
                        results.append((pos_code, definition, example, synonyms))
        except AttributeError:
            return []
        return results

    # ------------------------------------------------------------------
    # Learning: unknown words/terms are looked up online and remembered
    # ------------------------------------------------------------------

    def _ensure_learned(self) -> dict:
        """Load (once) the persistent learned dictionary."""
        if self._learned is None:
            data = None
            try:
                if os.path.exists(self._learn_path):
                    with open(self._learn_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError):
                data = None
            if not isinstance(data, dict) or not isinstance(data.get('entries'), dict):
                data = {'schema': 1, 'knowledge_version': __version__, 'entries': {}}
            data.setdefault('knowledge_version', __version__)
            data.setdefault('entries', {})
            self._learned = data
        return self._learned

    def _learned_entry(self, word_l: str) -> Optional[dict]:
        return self._ensure_learned()['entries'].get(word_l)

    def _save_learned(self) -> None:
        try:
            directory = os.path.dirname(self._learn_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._learn_path, 'w', encoding='utf-8') as f:
                json.dump(self._learned, f, ensure_ascii=False, indent=1)
        except OSError:
            pass  # read-only install: keep learning in memory only

    def learn(self, word: str, definitions: List[Tuple[str, str, str, List[str]]],
              source: str = 'web', url: str = '') -> bool:
        """
        Add a word or term with definitions to the learned dictionary and
        bump the knowledge version. `definitions` is a list of
        (pos, text, example, synonyms) tuples.
        """
        word = word.strip().lower()
        definitions = [d for d in definitions if d and d[1] and str(d[1]).strip()]
        if not word or not definitions:
            return False
        learned = self._ensure_learned()
        learned['entries'][word] = {
            'definitions': [
                {'pos': d[0], 'text': ' '.join(str(d[1]).split()),
                 'example': d[2] or '', 'synonyms': list(d[3] or [])}
                for d in definitions
            ],
            'source': source,
            'url': url,
            'learned_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        learned['knowledge_version'] = bump_patch(
            learned.get('knowledge_version', __version__))
        self._save_learned()
        return True

    def learn_from_summary(self, term: str, summary: str,
                           source: str = 'web', url: str = '') -> bool:
        """Learn a term (often multi-word) from a web summary text."""
        summary = ' '.join(str(summary or '').split()).strip()
        if len(summary) < 10:
            return False
        return self.learn(term, [('x', summary, '', [])], source=source, url=url)

    def forget(self, word: str) -> bool:
        """Remove a learned word or term from the learned dictionary."""
        learned = self._ensure_learned()
        if word.strip().lower() in learned['entries']:
            del learned['entries'][word.strip().lower()]
            self._save_learned()
            return True
        return False

    @property
    def knowledge_version(self) -> str:
        """The knowledge version - bumped each time a word is learned."""
        return self._ensure_learned().get('knowledge_version', __version__)

    @property
    def learned_words(self) -> List[str]:
        """All words and terms the AI has learned, sorted."""
        return sorted(self._ensure_learned()['entries'].keys())

    # ------------------------------------------------------------------
    # Spelling suggestions / autocomplete
    # ------------------------------------------------------------------

    def suggest(self, word: str, n: int = 5, max_distance: int = 2) -> List[str]:
        """
        Suggest correct spellings for a word using Damerau-Levenshtein
        distance against the most frequent English words.
        """
        word = word.strip().lower()
        if not word or self.is_word(word):
            return []

        freq = self._ensure_freq()
        best: List[Tuple[int, int, str]] = []  # (distance, -frequency, candidate)

        for candidate, count in freq.items():
            # Cheap pre-filter to keep this fast.
            if abs(len(candidate) - len(word)) > max_distance:
                continue
            if not (candidate[0] == word[0] or candidate[:2] == word[:2]):
                continue
            distance = _damerau_levenshtein(word, candidate, max_distance)
            if distance <= max_distance:
                best.append((distance, -count, candidate))

        best.sort()
        return [c for _, _, c in best[:n]]

    def complete(self, prefix: str, n: int = 10) -> List[str]:
        """Autocomplete a word prefix, most frequent first."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        freq = self._ensure_freq()
        matches = [(count, w) for w, count in freq.items() if w.startswith(prefix)]
        words = self._ensure_words()
        for w in words:
            if w.startswith(prefix) and w not in freq and len(w) > 2:
                matches.append((0, w))
        matches.sort(key=lambda item: -item[0])
        return [w for _, w in matches[:n]]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def format_entry(self, word: str, use_online: Optional[bool] = None,
                     max_senses: int = 5) -> str:
        """Format a dictionary entry as human-readable text."""
        entry = self.lookup(word, use_online=use_online)
        if not entry.found:
            if entry.suggestions:
                return (f"'{entry.word}' is not in the English dictionary. "
                        f"Did you mean: {', '.join(entry.suggestions)}?")
            return f"'{entry.word}' is not in the English dictionary."

        if not entry.definitions:
            return (f"'{entry.word}' is a valid English word, but I have no "
                    f"definition recorded for it.")

        lines = [f"📖 {entry.word} ({', '.join(entry.parts_of_speech)})"]
        for i, definition in enumerate(entry.definitions[:max_senses], 1):
            marker = f"  {i}. ({definition.pos_name}) "
            lines.append(marker + definition.text)
            if definition.example:
                lines.append(f'     e.g. "{definition.example}"')
            if definition.synonyms:
                lines.append(f"     synonyms: {', '.join(definition.synonyms[:5])}")
        return '\n'.join(lines)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            'word_list_size': len(self._ensure_words()),
            'defined_lemmas': len(self._ensure_entries()),
            'irregular_forms': len(self._ensure_exceptions()),
            'frequency_entries': len(self._ensure_freq()),
            'learned_words': len(self._ensure_learned()['entries']),
        }

    def random_word(self, min_len: int = 4, max_len: int = 12) -> str:
        import random
        words = self._ensure_words()
        for _ in range(200):
            w = random.choice(tuple(words)) if words else ''
            if min_len <= len(w) <= max_len and w.isalpha():
                return w
        return 'word'


# ----------------------------------------------------------------------
# Edit distance
# ----------------------------------------------------------------------

def _damerau_levenshtein(a: str, b: str, cutoff: int = 2) -> int:
    """Damerau-Levenshtein distance with an early-exit cutoff."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    if a == b:
        return 0
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        current = [i] + [0] * lb
        best_in_row = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + cost)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                value = min(value, prev2[j - 2] + 1)  # transposition
            current[j] = value
            best_in_row = min(best_in_row, value)
        if best_in_row > cutoff:
            return cutoff + 1
        prev2, prev = prev, current
    return prev[lb]
