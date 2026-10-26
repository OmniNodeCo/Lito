"""
GPT-2 style byte-level BPE tokenizer - implemented from scratch.

This is the tokenizer for the big (GGUF/llama.cpp) models. It is
deliberately GPT-2 compatible - same byte-to-unicode mapping, same
pre-tokenization regex, same merge format - so the vocabulary exports
directly into a GGUF file that llama.cpp, LM Studio and Ollama load
natively (tokenizer.ggml.model = "gpt2").

The smaller in-app model keeps using src/tokenizer.py; this module is
only imported by train_big.py / export_gguf.py and the tests.
"""

import json
import re
from collections import Counter, defaultdict

# The GPT-2 pre-tokenization regex. Matches llama.cpp's GPT2 splitter
# ('s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)) using stdlib
# re unicode classes: [^\W\d_] = \p{L}, \d = \p{Nd}, (?:[^\s\w]|_) = the
# "other" run (underscore counts as other, like \p{L}\p{N} exclusion).
# The trailing \s+ fallback (from the original GPT-2 encoder) guarantees
# findall never skips a character.
PRETOKENIZE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)

# Chat control tokens (ChatML markers) - exported as CONTROL tokens so
# llama.cpp always tokenizes them as single ids.
SPECIAL_TOKENS = ['<|im_start|>', '<|im_end|>']


def bytes_to_unicode():
    """GPT-2's reversible byte <-> unicode-character table."""
    bs = (list(range(ord('!'), ord('~') + 1))
          + list(range(ord('\xa1'), ord('\xac') + 1))
          + list(range(ord('\xae'), ord('\xff') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_MAP = bytes_to_unicode()
CHAR_MAP = {c: b for b, c in BYTE_MAP.items()}


class ByteBPE:
    """Byte-level BPE with GPT-2 semantics."""

    def __init__(self, vocab_size=4096):
        self.target_vocab_size = vocab_size
        # ids 0..255 are the byte tokens, then the special tokens
        self.vocab = [BYTE_MAP[b] for b in range(256)] + list(SPECIAL_TOKENS)
        self.merges = []  # list of (left, right) symbol pairs, in order
        self._ranks = {}
        self._cache = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, texts, verbose=True):
        """Learn merges from texts (an iterable of plain strings).

        Classic BPE with incremental bookkeeping: pairs are counted once
        and only the words containing the merged pair are rewritten, so
        training a ~25MB corpus stays fast.
        """
        # Word counts over pre-tokenized pieces. GPT-2 order: run the
        # regex on the raw text, then map each piece to byte-unicode
        # symbols (space -> 'GH'-style printable stand-ins).
        word_counts = Counter()
        for text in texts:
            word_counts.update(PRETOKENIZE.findall(text))

        words = []       # list of (symbols, count)
        for word, count in word_counts.items():
            syms = [BYTE_MAP[b] for b in word.encode('utf-8')]
            words.append((syms, count))

        # pair -> number of occurrences, and pair -> word indices
        counts = Counter()
        where = defaultdict(set)
        for i, (syms, cnt) in enumerate(words):
            for pair in zip(syms, syms[1:]):
                counts[pair] += cnt
                where[pair].add(i)

        num_merges = self.target_vocab_size - len(self.vocab)
        for step in range(num_merges):
            best, best_n = None, 0
            for pair, n in counts.items():
                if n > best_n:
                    best, best_n = pair, n
            if best is None or best_n < 2:
                break  # nothing frequent enough left to merge

            a, b = best
            merged = a + b
            self.merges.append(best)
            self.vocab.append(merged)

            # Rewrite only the words that contain the merged pair.
            for i in where.pop(best, ()):
                syms, cnt = words[i]
                out, j = [], 0
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        out.append(merged)
                        j += 2
                    else:
                        out.append(syms[j])
                        j += 1
                if out == syms:
                    continue  # stale entry; the pair is not in this word
                # keep counts exact: drop old pairs, add new pairs
                for pair in zip(syms, syms[1:]):
                    counts[pair] -= cnt
                    if counts[pair] <= 0:
                        del counts[pair]
                for pair in zip(out, out[1:]):
                    counts[pair] += cnt
                    where[pair].add(i)
                words[i] = (out, cnt)
            counts.pop(best, None)

            if verbose and (step + 1) % 500 == 0:
                print(f'  BPE merge {step + 1}/{num_merges} '
                      f'(vocab {len(self.vocab)})', flush=True)
        self._rebuild_ranks()
        self._cache.clear()
        return self

    def _rebuild_ranks(self):
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _bpe_word(self, word):
        """Apply the learned merges to one pre-tokenized word."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        syms = list(word)
        while len(syms) > 1:
            pairs = [(self._ranks.get((a, b), 1 << 30), a, b)
                     for a, b in zip(syms, syms[1:])]
            rank, a, b = min(pairs)
            if rank == 1 << 30:
                break
            merged = a + b
            out, j = [], 0
            while j < len(syms):
                if (j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b):
                    out.append(merged)
                    j += 2
                else:
                    out.append(syms[j])
                    j += 1
            syms = out
        self._cache[word] = syms
        return syms

    def encode(self, text, add_special_tokens=True):
        """Text -> token ids. Special (chat) tokens are matched first."""
        ids = []
        if add_special_tokens and self.special_ids:
            pattern = '(' + '|'.join(re.escape(s) for s in SPECIAL_TOKENS) + ')'
            for chunk in re.split(pattern, text):
                if not chunk:
                    continue
                if chunk in self.special_ids:
                    ids.append(self.special_ids[chunk])
                else:
                    ids.extend(self._encode_plain(chunk))
        else:
            ids = self._encode_plain(text)
        return ids

    def _encode_plain(self, text):
        ids = []
        token_to_id = self.token_to_id
        for word in PRETOKENIZE.findall(text):
            # GPT-2 order: regex on raw text, then byte-encode the piece
            word = ''.join(BYTE_MAP[b] for b in word.encode('utf-8'))
            for sym in self._bpe_word(word):
                tid = token_to_id.get(sym)
                if tid is not None:
                    ids.append(tid)
                else:
                    # Unreachable for well-formed vocabs (all 256 byte
                    # forms are in it) - safety net: raw byte ids.
                    for ch in sym:
                        ids.append(CHAR_MAP.get(ch, 0))
        return ids

    def decode(self, ids):
        out = []
        specials = self.id_to_special
        for tid in ids:
            special = specials.get(tid)
            if special is not None:
                out.append(special)
            else:
                out.append(self.vocab[tid])
        return _unicode_to_bytes(''.join(out))

    # ------------------------------------------------------------------
    # Vocab helpers
    # ------------------------------------------------------------------
    @property
    def special_ids(self):
        return {s: 256 + i for i, s in enumerate(SPECIAL_TOKENS)}

    @property
    def id_to_special(self):
        return {256 + i: s for i, s in enumerate(SPECIAL_TOKENS)}

    @property
    def token_to_id(self):
        return {t: i for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        data = {
            'vocab': self.vocab,
            'merges': [list(p) for p in self.merges],
            'specials': SPECIAL_TOKENS,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=len(data['vocab']))
        tok.vocab = data['vocab']
        tok.merges = [tuple(p) for p in data['merges']]
        tok._rebuild_ranks()
        tok._cache.clear()
        return tok


def _unicode_to_bytes(text):
    """Inverse of the GPT-2 byte->unicode mapping."""
    out = bytearray()
    for ch in text:
        b = CHAR_MAP.get(ch)
        if b is not None:
            out.append(b)
        # characters not produced by our vocab pass through unchanged
        else:
            out.extend(ch.encode('utf-8'))
    return out.decode('utf-8', errors='replace')
