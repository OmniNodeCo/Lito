"""
Byte-Pair Encoding (BPE) tokenizer built from scratch.
"""

import re
import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """A byte-pair encoding tokenizer."""

    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        '<SEP>': 4,
        '<MASK>': 5,
        '<USER>': 6,
        '<AI>': 7,
    }

    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._built = False

    def _get_pairs(self, word: List[str]) -> Counter:
        """Get frequency of adjacent pairs."""
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _tokenize_word(self, word: str) -> List[str]:
        """Split word into characters with end-of-word marker."""
        return list(word) + ['</w>']

    def build_vocab(self, texts: List[str]):
        """Build BPE vocabulary from texts."""
        print("Building BPE vocabulary...")

        # Start with character vocabulary + special tokens
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        next_id = len(self.SPECIAL_TOKENS)

        # Tokenize into words and get frequencies
        word_freqs = Counter()
        for text in texts:
            words = re.findall(r'\w+|[^\w\s]|\s+', text.lower())
            for word in words:
                word_freqs[word] += 1

        # Build initial character vocab
        all_chars = set()
        for word in word_freqs:
            for ch in word:
                all_chars.add(ch)
            all_chars.add('</w>')

        for ch in sorted(all_chars):
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                next_id += 1

        # Build word representations
        words_as_tokens = {}
        for word in word_freqs:
            words_as_tokens[word] = self._tokenize_word(word)

        # BPE merging
        num_merges = self.target_vocab_size - next_id
        self.merges = []

        for merge_step in range(max(0, num_merges)):
            # Count all pairs across all words
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                tokens = words_as_tokens[word]
                for i in range(len(tokens) - 1):
                    pair_freqs[(tokens[i], tokens[i + 1])] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = pair_freqs.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = next_id
                next_id += 1

            # Merge in all words
            for word in words_as_tokens:
                tokens = words_as_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                words_as_tokens[word] = new_tokens

            if (merge_step + 1) % 500 == 0:
                print(f"  BPE merge {merge_step + 1}/{num_merges}, vocab size: {next_id}")

        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._built = True
        print(f"Vocabulary built: {len(self.token_to_id)} tokens")

    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        tokens = self._tokenize_word(word)

        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not self._built:
            raise RuntimeError("Tokenizer not built. Call build_vocab first.")

        tokens = []
        if add_special_tokens:
            tokens.append(self.SPECIAL_TOKENS['<BOS>'])

        words = re.findall(r'\w+|[^\w\s]|\s+', text.lower())
        for word in words:
            bpe_tokens = self._apply_bpe(word)
            for t in bpe_tokens:
                tokens.append(self.token_to_id.get(t, self.SPECIAL_TOKENS['<UNK>']))

        if add_special_tokens:
            tokens.append(self.SPECIAL_TOKENS['<EOS>'])

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()
        for tid in token_ids:
            if tid in special_ids:
                continue
            token = self.id_to_token.get(tid, '<UNK>')
            tokens.append(token)

        text = ''.join(tokens)
        text = text.replace('</w>', '')
        return text

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def save(self, path: str):
        """Save tokenizer to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'token_to_id': self.token_to_id,
            'merges': self.merges,
            'target_vocab_size': self.target_vocab_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_to_id = data['token_to_id']
        # Convert string keys back to int for id_to_token
        self.token_to_id = {k: int(v) for k, v in self.token_to_id.items()}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = [(m[0], m[1]) for m in data['merges']]
        self.target_vocab_size = data['target_vocab_size']
        self._built = True
        print(f"Tokenizer loaded from {path} ({len(self.token_to_id)} tokens)")