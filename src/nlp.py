"""
Natural language processing built entirely from scratch - no NLP libraries.

This module gives Lito real sentence understanding instead of naive
keyword matching:

- Sentence segmentation (abbreviation aware)
- Tokenization
- Lemmatization (morphological rules + WordNet irregular forms)
- Negation detection ("I don't like X" should not be answered with X)
- Clause splitting for compound/complex sentences, so multi-part questions
  are answered part by part
- Question-type detection (what / why / how / who / when / where / which /
  yes-no / definition / spelling / synonym / search requests)
- Pronoun detection for coreference resolution across conversation turns
- Frequency-weighted keyword extraction and spelling analysis
- A BM25 ranking index for knowledge retrieval
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .dictionary import EnglishDictionary

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

STOPWORDS: Set[str] = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself let's me more most mustn't my myself no nor not
of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's
the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())

# Question words are stopwords for retrieval, but matter for intent.
QUESTION_WORDS = {'what', 'why', 'how', 'when', 'who', 'where', 'which', 'whom', 'whose'}

NEGATORS: Set[str] = frozenset("""
not no never nothing nobody none neither nor cannot without dont doesnt
didnt wont wouldnt cant couldnt shouldnt isnt arent wasnt werent hasnt
havent hadnt stop avoid lack
""".split())

PRONOUNS: Set[str] = frozenset(
    ['it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those',
     'he', 'she', 'him', 'her', 'one'])

# Words that carry intent rather than content ("define", "meaning", "synonym").
# They are excluded from clause focus so retrieval keys on the actual subject.
META_WORDS: Set[str] = frozenset(
    ['define', 'definition', 'meaning', 'mean', 'means', 'meant', 'explain',
     'synonym', 'synonyms', 'spell', 'spelling', 'dictionary', 'word', 'words',
     'search', 'google', 'find', 'look', 'tell', 'another', 'different',
     'example', 'examples', 'sentence', 'phrase', 'term', 'give', 'show',
     'list', 'name', 'describe', 'answer', 'question', 'call', 'called'])

# Conjunctions on which complex/compound sentences are split into clauses.
CLAUSE_SPLITTERS = frozenset(
    ['because', 'although', 'though', 'while', 'whereas', 'unless', 'since',
     'if', 'when', 'after', 'before', 'until', 'and', 'but', 'or', 'yet',
     'so', 'however', 'therefore'])

ABBREVIATIONS = frozenset(
    ['mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g',
     'i.e', 'approx', 'fig', 'no', 'inc', 'ltd', 'co', 'jan', 'feb', 'mar',
     'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
     'p.m', 'a.m', 'u.s', 'u.k', 'ph.d', 'b.c', 'a.d'])

_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sa-zA-Z0-9]")

# Patterns that signal a definition request. Group 1 captures the target.
_DEFINITION_EXTRACTORS = [
    # "what is love", "what's the meaning of life", "which is the biggest city"
    re.compile(r"^(?:what|which)(?:'s\s+|\s+is\s+|\s+are\s+|\s+was\s+|\s+were\s+)"
               r"(?:a|an|the)?\s*(?:word|term|phrase|concept|topic|subject)?\s*(.+?)\s*\??\s*$", re.I),
    # "define serendipity", "definition of gravity", "meaning of life"
    re.compile(r"^(?:define|definition of|meaning of|meanings of)\s+"
               r"(?:the\s+word\s+|the\s+term\s+|the\s+phrase\s+)?(.+?)\s*\??\s*$", re.I),
    # "what does serendipity mean"
    re.compile(r"^what\s+(?:does|do)\s+(?:the\s+word\s+|the\s+term\s+)?(.+?)\s+mean\b.*$", re.I),
    # "tell me what love is", "do you know what gravity is"
    re.compile(r"\bwhat\s+(?:(?:a|an|the)\s+)?(?!does\b|do\b|did\b)(.+?)\s+(?:is|are|means?)\b", re.I),
    # "what do you call a group of crows"
    re.compile(r"^what\s+(?:do|does)\s+(?:you|we|they)\s+call\s+(.+?)\s*\??\s*$", re.I),
]

# Search-style requests: "search for X", "look up X", "google X".
_SEARCH_PATTERNS = [
    re.compile(r"^(?:please\s+)?(?:search|google|look\s?up|find|wiki)\s+"
               r"(?:for\s+|up\s+|about\s+|on\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"^(?:can you|could you|please)\s+(?:search|google|look\s?up|find)\s+"
               r"(?:for\s+|up\s+|about\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"\bsearch the (?:web|internet|net)\s+(?:for\s+)?(.+?)\s*\??\s*$", re.I),
]

# Patterns that signal spelling questions.
_SPELL_PATTERNS = [
    re.compile(r"^(?:how (?:do|would|did) (?:you|i|we) spell|how is .* spelled|"
               r"spell(?: out)?)\b", re.I),
    re.compile(r"\bcorrect spelling of\b", re.I),
]

# Patterns for "is X a (real/valid/english) word?"
IS_WORD_PATTERNS = [
    re.compile(r"^is\s+[\"']?(.+?)[\"']?\s+"
               r"(?:(?:a|an|real|valid|actual|proper|correct|english)\s+)*"
               r"(?:english\s+)?word\??\s*$", re.I),
    re.compile(r"\b(?:is|was)\s+[\"']?(.+?)[\"']?\s+in the (?:english )?dictionary\b", re.I),
    re.compile(r"^(?:does|do)\s+the\s+word\s+[\"']?(.+?)[\"']?\s+exist\b", re.I),
]

# Synonym requests: "another word for X", "synonyms of X".
_SYNONYM_PATTERNS = [
    re.compile(r"^(?:another|other)\s+word\s+(?:for|to)\s+(.+?)\s*\??\s*$", re.I),
    re.compile(r"^synonyms?\s+(?:of|for)\s+(.+?)\s*\??\s*$", re.I),
    re.compile(r"^what(?:'s| is| are)\s+(?:the\s+)?synonyms?\s+(?:of|for)\s+(.+?)\s*\??\s*$", re.I),
    re.compile(r"^what(?:'s| is| are)\s+(?:a|an|some)?\s*(?:other|different)\s+word\s+"
               r"(?:for|to)\s+(.+?)\s*\??\s*$", re.I),
    re.compile(r"^what\s+(?:does\s+)?(?:words?|terms?)\s+mean the same as\s+(.+?)\s*\??\s*$", re.I),
]

_PRONOUN_COREF_RE = re.compile(
    r"\b(it|this|that|they|them|these|those|he|she|him|her)\b(?!['\u2019][a-z])", re.I)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class Clause:
    """One unit of meaning: a question, statement or request."""
    text: str
    is_question: bool = False
    question_type: str = ''          # what/why/how/.../definition/spell/isword/synonyms/search
    negated: bool = False
    focus: List[str] = field(default_factory=list)        # content lemmas, negation-aware
    entities: List[str] = field(default_factory=list)     # proper-noun candidates
    quote: str = ''                                       # quoted span, if any
    definition_target: str = ''                           # extracted definition target
    search_query: str = ''                                # extracted query for search intents


@dataclass
class Analysis:
    """Full understanding of one user utterance."""
    original: str
    cleaned: str
    sentences: List[str] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    pronouns: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)
    spelling_fixes: Dict[str, str] = field(default_factory=dict)
    resolved_text: str = ''           # input with pronouns replaced by the last topic
    has_coreference: bool = False

    @property
    def is_question(self) -> bool:
        return any(c.is_question for c in self.clauses)

    @property
    def question_type(self) -> str:
        for c in self.clauses:
            if c.question_type:
                return c.question_type
        return ''


# ----------------------------------------------------------------------
# Tokenization & segmentation
# ----------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize into words, numbers and punctuation."""
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, tolerating common abbreviations."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # Protect abbreviation dots: "Dr." -> "Dr<ABBR>" (case preserved)
    protected = text
    for abbr in ABBREVIATIONS:
        protected = re.sub(
            rf"\b{re.escape(abbr)}\.(?=\s|$)",
            lambda m: m.group(0).replace('.', '<ABBR>'), protected, flags=re.I)
    # Protect decimals like 3.14
    protected = re.sub(r'(\d)\.(\d)', r'\1<ABBR>\2', protected)

    parts = re.split(r'(?<=[.!?])\s+', protected)
    sentences = [p.replace('<ABBR>', '.').strip() for p in parts if p.strip()]

    # Re-attach pure punctuation fragments.
    merged = []
    for s in sentences:
        if merged and len(s) <= 2 and not re.search(r'[a-zA-Z0-9]', s):
            merged[-1] += s
        else:
            merged.append(s)
    return merged


# ----------------------------------------------------------------------
# Lemmatization
# ----------------------------------------------------------------------

class Lemmatizer:
    """
    Morphological lemmatizer. Uses WordNet's irregular-form table first,
    then regular suffix rules, validating candidates against the word list.
    """

    _SUFFIX_RULES = [
        ('ies', 'y'),      # studies -> study
        ('ied', 'y'),      # carried -> carry
        ('ying', 'ie'),    # lying -> lie
        ('ing',),          # running -> run (double consonant), making -> make
        ('ed',),           # stopped -> stop (double consonant), walked -> walk
        ('est', ''),       # greatest -> great
        ('er', ''),        # greater -> great
        ('ly', ''),        # quickly -> quick
        ('es', ''),        # boxes -> box
        ('s', ''),         # cats -> cat
    ]

    def __init__(self, dictionary: Optional[EnglishDictionary] = None):
        self.dictionary = dictionary or EnglishDictionary(online=False)

    def _known(self, word: str) -> bool:
        return self.dictionary.is_word(word)

    def _is_base_lemma(self, word: str) -> bool:
        """True when WordNet lists the word as a base lemma (e.g. 'species')."""
        return bool(self.dictionary.pos_of(word))

    def lemma(self, word: str) -> str:
        """Return the base form of a word."""
        word = word.lower().strip()
        if not word or len(word) <= 2:
            return word
        if word.endswith("'s"):
            word = word[:-2]

        # 1. Irregular forms (wolves -> wolf, better -> good, went -> go)
        for _pos, base in self.dictionary.irregular_bases(word):
            return base

        # 2. Verb inflections: prefer the verb base (running -> run,
        #    carried -> carry) even though -ing/-ed forms can be nouns too.
        if word.endswith('ing') or word.endswith('ed'):
            for candidate in self._verb_base_candidates(word):
                if self._known(candidate) and 'v' in self.dictionary.pos_of(candidate):
                    return candidate

        # 3. Word is itself a base form in WordNet (species, data, physics)
        if self._is_base_lemma(word):
            return word

        # 4. Regular suffix rules (studies -> study, cats -> cat)
        for rule in self._SUFFIX_RULES:
            suffix = rule[0]
            if not word.endswith(suffix) or len(word) - len(suffix) < 2:
                continue
            stem = word[:len(word) - len(suffix)]
            candidates = []
            if len(rule) > 1:
                candidates.append(stem + rule[1])
            else:
                # stripping -ing/-ed/-es/-s: try stem, stem+e, and doubled
                # consonant reduction (running -> runn -> run)
                candidates.append(stem)
                candidates.append(stem + 'e')
                if len(stem) >= 2 and stem[-1] == stem[-2]:
                    candidates.append(stem[:-1])
            for candidate in candidates:
                if self._known(candidate):
                    return candidate

        # 5. Known word with no rule applying (necessary, coffee)
        if self._known(word):
            return word

        return word

    @staticmethod
    def _verb_base_candidates(word: str) -> List[str]:
        """Candidate verb bases for an -ing/-ed/-ied form."""
        if word.endswith('ied'):
            return [word[:-3] + 'y']                       # carried -> carry
        if word.endswith('ying'):
            return [word[:-4] + 'ie']                      # lying -> lie
        if word.endswith('ing') or word.endswith('ed'):
            stem = word[:-3]
            candidates = [stem, stem + 'e']                # walk -> walk, mak -> make
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                candidates.append(stem[:-1])               # runn -> run
            return candidates
        return []


# ----------------------------------------------------------------------
# Lightweight part-of-speech hints
# ----------------------------------------------------------------------

_ADJ_SUFFIXES = ('ous', 'ful', 'ive', 'able', 'ible', 'al', 'ic', 'ish', 'less')
_NOUN_SUFFIXES = ('tion', 'ment', 'ness', 'ity', 'ship', 'hood', 'ism', 'ist', 'ance', 'ence')
_ADV_SUFFIXES = ('ly', 'wise')


def pos_of_token(token: str, is_capitalized: bool, dictionary: EnglishDictionary) -> str:
    """
    Guess the part of speech of a token ('n', 'v', 'a', 'r', 'p' proper noun).
    Uses WordNet POS data plus suffix and capitalization heuristics.
    """
    lower = token.lower()
    if is_capitalized:
        return 'p'
    wordnet_pos = dictionary.pos_of(lower)
    if wordnet_pos:
        if 'n' in wordnet_pos:
            return 'n'
        if 'v' in wordnet_pos:
            return 'v'
        if 'a' in wordnet_pos:
            return 'a'
        return 'r'
    if lower.endswith(_ADV_SUFFIXES):
        return 'r'
    if lower.endswith(_ADJ_SUFFIXES):
        return 'a'
    if lower.endswith(_NOUN_SUFFIXES):
        return 'n'
    return 'x'


# ----------------------------------------------------------------------
# Negation handling
# ----------------------------------------------------------------------

def negated_scope(tokens: List[str]) -> Set[int]:
    """
    Return token indices inside a negation scope: from a negator until the
    next punctuation boundary (max 4 tokens of scope).
    """
    negated: Set[int] = set()
    scope = 0
    for i, token in enumerate(tokens):
        lower = token.lower().replace("'", '')
        if lower in NEGATORS:
            scope = 4
            continue
        if not re.search(r'[a-zA-Z0-9]', token):   # punctuation ends scope
            scope = 0
            continue
        if scope > 0:
            negated.add(i)
            scope -= 1
    return negated


# ----------------------------------------------------------------------
# The analyzer
# ----------------------------------------------------------------------

class Analyzer:
    """Turns raw user input into a structured Analysis."""

    def __init__(self, dictionary: Optional[EnglishDictionary] = None,
                 spellcheck: bool = True):
        self.dictionary = dictionary or EnglishDictionary(online=False)
        self.lemmatizer = Lemmatizer(self.dictionary)
        self.spellcheck = spellcheck

    # -- public API -------------------------------------------------------

    def analyze(self, text: str, last_topic: str = '') -> Analysis:
        """Analyze an utterance. `last_topic` enables coreference resolution."""
        cleaned = re.sub(r'\s+', ' ', text).strip()
        analysis = Analysis(original=text, cleaned=cleaned)

        # Spelling analysis on raw words first (so downstream sees fixes)
        if self.spellcheck:
            for word in tokenize(cleaned):
                if (word.isalpha() and len(word) > 3 and word.islower()
                        and not self.dictionary.is_word(word)):
                    suggestions = self.dictionary.suggest(word, n=1)
                    if suggestions:
                        analysis.spelling_fixes[word] = suggestions[0]
                    elif word not in analysis.unknown_words:
                        analysis.unknown_words.append(word)

        # Coreference: swap pronouns for the previous topic
        resolved = cleaned
        if last_topic:
            resolved, used = self._resolve_coreference(cleaned, last_topic)
            analysis.has_coreference = used
        analysis.resolved_text = resolved

        # Sentence segmentation then clause extraction
        analysis.sentences = split_sentences(resolved)
        for sentence in analysis.sentences:
            for clause_text in self._split_clauses(sentence):
                clause = self._analyze_clause(clause_text)
                if clause and (clause.focus or clause.question_type
                               or not clause.is_question):
                    analysis.clauses.append(clause)

        if not analysis.clauses:
            analysis.clauses = [self._analyze_clause(cleaned)]

        # Global keyword list (ordered, deduplicated)
        seen = set()
        for clause in analysis.clauses:
            for kw in clause.focus:
                if kw not in seen:
                    seen.add(kw)
                    analysis.keywords.append(kw)
            analysis.entities.extend(clause.entities)

        analysis.pronouns = [t.lower() for t in tokenize(cleaned)
                             if t.lower() in PRONOUNS]
        return analysis

    # -- internals ----------------------------------------------------------

    def _resolve_coreference(self, text: str, topic: str) -> Tuple[str, bool]:
        """Replace standalone pronouns with the conversation's last topic."""
        if not _PRONOUN_COREF_RE.search(text):
            return text, False
        resolved = _PRONOUN_COREF_RE.sub(topic.strip(), text)
        return resolved, True

    def _split_clauses(self, sentence: str) -> List[str]:
        """
        Split a compound/complex sentence into meaningful clause units so
        each part can be handled separately.
        """
        pieces = [p.strip() for p in sentence.split(';') if p.strip()] or [sentence]

        clauses: List[str] = []
        for piece in pieces:
            # Drop a leading conjunction ("and what about X" -> "what about X")
            piece = re.sub(r'^(?:and|or|but|so|also|then)\s+', '', piece, flags=re.I)
            # Split on " and ", " but ", " because ", ... when both sides
            # are substantial.
            tokens = piece.split()
            split_points = []
            for i, tok in enumerate(tokens):
                lower = tok.lower().strip(',;:')
                if lower not in CLAUSE_SPLITTERS:
                    continue
                left_words = sum(1 for w in tokens[:i] if re.search(r'[a-zA-Z]', w))
                right_words = sum(1 for w in tokens[i + 1:] if re.search(r'[a-zA-Z]', w))
                if left_words >= 1 and right_words >= 2:
                    split_points.append(i)
            if not split_points:
                clauses.append(piece)
                continue
            bounds = [-1] + split_points + [len(tokens)]
            for a, b in zip(bounds, bounds[1:]):
                part = ' '.join(tokens[a + 1:b]).strip(' ,;:')
                # Drop a leading conjunction left over from the split
                part = re.sub(r'^(?:and|or|but|so|also|then)\s+', '', part, flags=re.I)
                if part and re.search(r'[a-zA-Z0-9]', part):
                    clauses.append(part)

        # Drop clause fragments with no content at all
        clauses = [c for c in clauses if self._content_tokens(c) or '?' in c]
        return (clauses or [sentence])[:4]

    def _content_tokens(self, text: str) -> List[str]:
        return [t for t in tokenize(text)
                if re.search(r'[a-zA-Z]', t) and t.lower() not in STOPWORDS
                and t.lower() not in QUESTION_WORDS]

    def _analyze_clause(self, text: str) -> Optional[Clause]:
        text = text.strip()
        if not text:
            return None
        tokens = tokenize(text)
        lowered = [t.lower() for t in tokens]

        clause = Clause(text=text)
        clause.is_question = self._is_question(text, lowered)
        clause.quote = self._extract_quote(text)

        # Entities: capitalized words (not at the very start of statements,
        # where they may just be capitalized sentence openers)
        for i, tok in enumerate(tokens):
            if re.match(r'^[A-Z][a-z]+', tok) and (i > 0 or clause.is_question):
                if tok.lower() not in QUESTION_WORDS and tok.lower() not in STOPWORDS \
                        and tok.lower() not in ('i', 'ai', 'ok', 'im'):
                    clause.entities.append(tok)

        # Lemmatized content words, respecting negation scope
        negated_indices = negated_scope(tokens)
        clause.negated = bool(negated_indices)
        for i, tok in enumerate(tokens):
            if i in negated_indices:
                continue
            lower = tok.lower()
            if lower in STOPWORDS or lower in QUESTION_WORDS:
                continue
            if not re.search(r'[a-zA-Z]', tok):
                continue
            lemma = self.lemmatizer.lemma(lower)
            if lemma not in ('be', 'do', 'have', 'will', 'just', 'please',
                             'would', 'could', 'shall', 'may', 'might', 'let') \
                    and lemma not in META_WORDS:
                if lemma not in clause.focus:
                    clause.focus.append(lemma)

        # Intent classification
        clause.question_type = self._question_type(text, lowered)
        clause.definition_target = self._extract(text, clause.quote, clause.question_type)

        # Search intent: extract the query part
        for pattern in _SEARCH_PATTERNS:
            m = pattern.search(text)
            if m and m.group(1) and len(m.group(1).strip()) > 1:
                clause.question_type = 'search'
                clause.search_query = m.group(1).strip(' ?."\'')
                break

        return clause

    def _is_question(self, text: str, lowered: List[str]) -> bool:
        if text.rstrip().endswith('?'):
            return True
        aux = {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could',
               'will', 'would', 'shall', 'should', 'have', 'has', 'had', 'may', 'might'}
        if len(lowered) >= 2 and lowered[0] in aux:
            return True
        if lowered and lowered[0] in QUESTION_WORDS:
            return True
        return False

    def _question_type(self, text: str, lowered: List[str]) -> str:
        for pattern in _SPELL_PATTERNS:
            if pattern.search(text):
                return 'spell'
        for pattern in IS_WORD_PATTERNS:
            if pattern.search(text):
                return 'isword'
        for pattern in _SYNONYM_PATTERNS:
            if pattern.search(text):
                return 'synonyms'
        for pattern in _DEFINITION_EXTRACTORS:
            if pattern.search(text):
                return 'definition'
        if not lowered:
            return ''
        first = lowered[0]
        if first in ('what', 'why', 'how', 'when', 'who', 'where', 'which', 'whom', 'whose'):
            return first
        for tok in lowered[:4]:
            if tok in ('why', 'how', 'when', 'who', 'where', 'which'):
                return tok
        if self._is_question(text, lowered) and '?' in text:
            return 'yesno'
        return ''

    def _extract(self, text: str, quote: str, question_type: str = '') -> str:
        """Extract the target word/phrase for definition & synonym intents."""
        if quote:
            return quote.strip()
        # Try the pattern family that matches the detected intent first, so
        # e.g. "what is another word for happy" extracts "happy" (not
        # "nother word for happy" from the generic definition pattern).
        if question_type == 'synonyms':
            patterns = _SYNONYM_PATTERNS + _DEFINITION_EXTRACTORS
        else:
            patterns = _DEFINITION_EXTRACTORS + _SYNONYM_PATTERNS
        for pattern in patterns:
            m = pattern.search(text)
            if m and m.group(1):
                target = m.group(1).strip(' ?."\'!:;')
                # strip leading articles
                target = re.sub(r'^(?:a|an|the)\s+', '', target, flags=re.I)
                if target:
                    return target
        return ''

    def _extract_quote(self, text: str) -> str:
        m = re.search(r'["\']([^"\']{2,60})["\']', text)
        if m:
            return m.group(1)
        m = re.search(r'\bmean by\s+([a-zA-Z][a-zA-Z\s-]{2,40})\??\s*$', text, re.I)
        if m:
            return m.group(1).strip()
        return ''


# ----------------------------------------------------------------------
# BM25 knowledge-ranking index
# ----------------------------------------------------------------------

class BM25Index:
    """
    Okapi BM25 ranking over a set of documents. Used to retrieve the most
    relevant piece of knowledge for a query instead of naive word overlap.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.doc_terms: List[Counter] = []
        self.df: Counter = Counter()
        self.total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        terms = self._tokenize_for_index(text)
        if not terms:
            return
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(terms))
        counts = Counter(terms)
        self.doc_terms.append(counts)
        for term in counts:
            self.df[term] += 1
        self.total_len += len(terms)

    def __len__(self) -> int:
        return len(self.doc_ids)

    @staticmethod
    def _tokenize_for_index(text: str) -> List[str]:
        return [t.lower() for t in _WORD_RE.findall(text)
                if re.search(r'[a-zA-Z0-9]', t) and t.lower() not in STOPWORDS]

    def search(self, query: str, n: int = 5,
               term_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, float]]:
        """
        Return [(doc_id, score, coverage)] best-first, where coverage is the
        fraction of distinct query terms that matched the document. Coverage
        distinguishes real topical matches (most terms hit) from coincidental
        single-word hits. `term_weights` optionally up/down-weights individual
        query terms (e.g. rare, topical words count more than common verbs).
        """
        if not self.doc_ids:
            return []
        query_terms = [t.lower() for t in _WORD_RE.findall(query)
                       if re.search(r'[a-zA-Z0-9]', t) and t.lower() not in STOPWORDS]
        if not query_terms:
            return []

        n_docs = len(self.doc_ids)
        avg_len = self.total_len / n_docs
        results: Dict[str, List[float]] = {}
        query_counter = Counter(query_terms)
        n_query_terms = len(query_counter)
        # Weighted coverage: how much of the query's total weight actually
        # matched. A doc that only matches a low-value common word ("work")
        # while missing the topical word ("vaccines") should not rank as a
        # confident match even if 1 of 2 terms hit.
        if term_weights:
            query_counter = Counter({term: count * term_weights.get(term, 1.0)
                                     for term, count in query_counter.items()})
            total_weight = sum(term_weights.get(t, 1.0) for t in query_counter)
        else:
            total_weight = float(n_query_terms)

        for idx, doc_counts in enumerate(self.doc_terms):
            score = 0.0
            matched_weight = 0.0
            dl = self.doc_len[idx]
            norm = self.k1 * (1 - self.b + self.b * dl / avg_len)
            for term, q_weight in query_counter.items():
                tf = doc_counts.get(term, 0)
                if tf == 0:
                    continue
                matched_weight += term_weights.get(term, 1.0) if term_weights else 1.0
                df = self.df[term]
                idf = max(0.1, math.log(1 + (n_docs - df + 0.5) / (df + 0.5)))
                score += q_weight * idf * tf * (self.k1 + 1) / (tf + norm)
            if matched_weight:
                results[self.doc_ids[idx]] = [score, matched_weight / total_weight]

        ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
        return [(doc_id, vals[0], vals[1]) for doc_id, vals in ranked[:n]]
