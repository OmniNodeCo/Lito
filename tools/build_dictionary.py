#!/usr/bin/env python3
"""
Build the offline English dictionary data files from WordNet 3.0 (WNDB format).

This script parses the original WordNet database files (index.*, data.*, *.exc)
entirely from scratch - no NLTK or other dependencies required - and produces
compact, gzipped data files used by src/dictionary.py:

  data/wordnet_dictionary.jsonl.gz  - one JSON line per lemma with definitions,
                                      synonyms and an example sentence per POS.
  data/wordnet_exceptions.tsv       - irregular morphology (inflected -> base).

To (re)generate the data:
  1. Download WordNet 3.0 database files, e.g. clone nltk/nltk_data and
     unzip packages/corpora/wordnet.zip, or fetch from
     https://wordnetcode.princeton.edu/wn3.0.tar.gz
  2. python tools/build_dictionary.py --wordnet-dir /path/to/wordnet

WordNet 3.0 license: Princeton University "WordNet 3.0 ... this software and
database is being provided by the copyright holders ... for any purpose
without fee or royalty" (see data/WORDNET_LICENSE).
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

# WordNet part-of-speech file suffixes and their single-letter codes.
POS_FILES = {'noun': 'n', 'verb': 'v', 'adj': 'a', 'adv': 'r'}

# Satellite adjectives in data.adj are marked 's'; map them onto 'a'.
SS_TYPE_TO_POS = {'n': 'n', 'v': 'v', 'a': 'a', 's': 'a', 'r': 'r'}

MAX_DEFS = 4       # definitions kept per (lemma, pos)
MAX_SYNS = 8       # synonyms kept per (lemma, pos)
MAX_EXAMPLES = 1   # example sentences kept per (lemma, pos)

_EXAMPLE_RE = re.compile(r'"([^"]+)"')


def parse_index_file(path):
    """
    Parse a WordNet index.* file.

    Each non-header line looks like:
        lemma pos synset_cnt p_cnt [ptr_symbol...] sense_cnt tagsense_cnt synset_offset...
    where synset offsets are ordered from the most frequent sense to the least.
    Returns {lemma: [synset_offset, ...]}.
    """
    lemma_offsets = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith(' '):
                continue  # license header lines are indented
            parts = line.split()
            if len(parts) < 4 or parts[1] not in ('n', 'v', 'a', 'r', 's'):
                continue
            lemma = parts[0]
            try:
                synset_cnt = int(parts[2])
                p_cnt = int(parts[3])
            except ValueError:
                continue
            # After the p_cnt pointer symbols come sense_cnt, tagsense_cnt,
            # then the synset offsets.
            idx = 4 + p_cnt + 2
            offsets = parts[idx:idx + synset_cnt]
            if len(offsets) != synset_cnt:
                continue
            lemma_offsets[lemma] = offsets
    return lemma_offsets


def parse_data_file(path):
    """
    Parse a WordNet data.* file.

    Each non-header line looks like:
        synset_offset lex_filenum ss_type w_cnt word lex_id [word lex_id...]
        p_cnt [ptr...] [frames...] | gloss [; "example sentence"]
    Returns {synset_offset: (ss_type, [words...], gloss)}.
    """
    synsets = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith(' '):
                continue
            if ' | ' not in line:
                continue
            head, gloss = line.split(' | ', 1)
            gloss = gloss.strip()
            parts = head.split()
            if len(parts) < 4:
                continue
            offset, _lex_filenum, ss_type, w_cnt_hex = parts[0], parts[1], parts[2], parts[3]
            try:
                w_cnt = int(w_cnt_hex, 16)
            except ValueError:
                continue
            words = []
            idx = 4
            for _ in range(w_cnt):
                if idx >= len(parts):
                    break
                words.append(parts[idx])
                idx += 2  # skip the lex_id
            synsets[offset] = (ss_type, words, gloss)
    return synsets


def split_gloss(gloss):
    """Split a WordNet gloss into (definition, [examples])."""
    examples = _EXAMPLE_RE.findall(gloss)
    definition = _EXAMPLE_RE.sub('', gloss)
    definition = definition.replace(' ;', ';').replace('; ', ' ').replace(';', ' ').strip()
    # Drop usage attribution fragments like "--Albert Einstein"
    definition = re.sub(r'\s*--[^;|"]+', ' ', definition)
    definition = re.sub(r'\s+', ' ', definition).strip(' ;')
    return definition, examples


def main():
    parser = argparse.ArgumentParser(description='Build dictionary data from WordNet WNDB files')
    parser.add_argument('--wordnet-dir', required=True,
                        help='Directory containing WordNet index.*/data.* files')
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    args = parser.parse_args()

    wn_dir = args.wordnet_dir
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(os.path.join(wn_dir, 'index.noun')):
        print(f"ERROR: {wn_dir} does not contain WordNet index.*/data.* files")
        sys.exit(1)

    # ---- Parse synsets -------------------------------------------------
    print('Parsing synset databases...')
    # pos -> offset -> (ss_type, words, gloss)
    synsets = {pos: parse_data_file(os.path.join(wn_dir, f'data.{name}'))
               for name, pos in POS_FILES.items()}
    total_synsets = sum(len(v) for v in synsets.values())
    print(f'  {total_synsets:,} synsets parsed')

    # ---- Parse index files and build per-lemma entries ------------------
    print('Building lemma entries...')
    entries = {}  # lemma -> {pos: {...}}
    for name, pos in POS_FILES.items():
        index = parse_index_file(os.path.join(wn_dir, f'index.{name}'))
        for lemma, offsets in index.items():
            pos_entry = {'d': [], 's': [], 'e': []}
            seen_defs = set()
            syn_candidates = []
            for offset in offsets:  # offsets are sense-ordered (most frequent first)
                syn = synsets[pos].get(offset)
                if syn is None:
                    continue
                _ss_type, words, gloss = syn
                definition, examples = split_gloss(gloss)
                if definition and definition not in seen_defs and len(pos_entry['d']) < MAX_DEFS:
                    seen_defs.add(definition)
                    pos_entry['d'].append(definition)
                    if len(pos_entry['e']) < MAX_EXAMPLES and examples:
                        pos_entry['e'].append(examples[0])
                # Collect synonyms from the primary senses first.
                if len(syn_candidates) < MAX_SYNS:
                    for w in words:
                        w = w.split('(')[0]      # strip markers like big(a)
                        if w and w != lemma and w not in syn_candidates:
                            syn_candidates.append(w)
            pos_entry['s'] = syn_candidates[:MAX_SYNS]
            if pos_entry['d']:
                entries.setdefault(lemma, {})[pos] = pos_entry
        print(f'  {name}: {len(index):,} lemmas indexed')

    # ---- Write the compact JSONL dictionary ----------------------------
    dict_path = os.path.join(out_dir, 'wordnet_dictionary.jsonl.gz')
    print(f'Writing {dict_path} ...')
    with gzip.open(dict_path, 'wt', encoding='utf-8') as f:
        for lemma in sorted(entries):
            f.write(json.dumps({'w': lemma, 'p': entries[lemma]},
                               ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'  {len(entries):,} lemmas written')

    # ---- Write irregular morphology exceptions --------------------------
    exc_path = os.path.join(out_dir, 'wordnet_exceptions.tsv')
    print(f'Writing {exc_path} ...')
    count = 0
    with open(exc_path, 'w', encoding='utf-8') as f:
        for name, pos in POS_FILES.items():
            exc_file = os.path.join(wn_dir, f'{name}.exc')
            if not os.path.exists(exc_file):
                continue
            with open(exc_file, 'r', encoding='utf-8') as ef:
                for line in ef:
                    parts = line.split()
                    if len(parts) >= 2:
                        f.write(f'{pos}\t{parts[0]}\t{parts[1]}\n')
                        count += 1
    print(f'  {count:,} irregular forms written')

    size = os.path.getsize(dict_path)
    print(f'\nDone! wordnet_dictionary.jsonl.gz is {size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
