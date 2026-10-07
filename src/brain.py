"""
The AI Brain - combines all components into an intelligent system.

Pipeline for every message:
1. Understand the sentence (src/nlp.py): segmentation, clause splitting,
   lemmatization, negation, question type, coreference, spelling.
2. Route each clause: small talk, dictionary lookups (definition / spelling /
   synonyms / word validity), search requests, knowledge retrieval.
3. Retrieve knowledge with BM25 ranking over the built-in knowledge base
   and training corpus.
4. Fall back to live web search (Wikipedia + DuckDuckGo) when local
   knowledge is not confident.
5. Optionally blend in neural generation from the trained model.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import TransformerLM
from .tokenizer import BPETokenizer
from .dataset import TrainingData
from .dictionary import EnglishDictionary, WordEntry
from .nlp import (Analyzer, Clause, Analysis, BM25Index, split_sentences,
                  tokenize, IS_WORD_PATTERNS)
from .search import WebSearch, SearchResult

# BM25 score at/above which local knowledge is considered a confident match
# (combined with query-term coverage - see _retrieve_knowledge).
KNOWLEDGE_CONFIDENCE_THRESHOLD = 4.0
# A very high BM25 score is confident on its own, regardless of coverage.
KNOWLEDGE_STRONG_SCORE = 8.0
# Below this, even a top match is treated as weak (search may be tried first).
KNOWLEDGE_WEAK_THRESHOLD = 2.0


class AIBrain:
    """
    The central AI system that combines:
    - Neural network language model
    - Pattern matching and small talk
    - The full English dictionary (WordNet + 370k word list)
    - BM25 knowledge retrieval
    - Live web search
    - Conversation management with coreference tracking
    """

    def __init__(self, model_dir: str = 'checkpoints', search_enabled: bool = True,
                 dictionary_online: bool = True):
        self.model_dir = model_dir
        self.model: Optional[TransformerLM] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.knowledge_base = self._build_knowledge_base()
        self.loaded = False

        # New capabilities
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(_root, 'data')
        self.dictionary = EnglishDictionary(data_dir=data_dir, online=dictionary_online)
        self.analyzer = Analyzer(self.dictionary, spellcheck=True)
        self.search = WebSearch(cache_dir=os.path.join(data_dir, 'cache'),
                                enabled=search_enabled)
        self.last_topic = ''
        self._index: Optional[BM25Index] = None
        self._last_knowledge: List[Tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------

    def _build_knowledge_base(self) -> Dict:
        """Build a structured knowledge base for retrieval."""
        kb = {
            'greetings': [
                "Hello! I'm SmartAI, an artificial intelligence built from scratch. How can I help you?",
                "Hi there! I'm ready to help you with questions, conversation, or just chatting.",
                "Hey! Nice to meet you. I'm an AI assistant. What would you like to talk about?",
            ],
            'identity': [
                "I'm SmartAI, a neural network-based AI built entirely from scratch in Python. "
                "I use a transformer architecture with self-attention, trained on a knowledge corpus.",
                "I'm SmartAI, an artificial intelligence created without using any pre-trained models. "
                "My neural network was designed and trained from the ground up.",
            ],
            'capabilities': [
                "I can answer questions, have conversations, explain concepts, define any English "
                "word from my built-in dictionary, search the web for current facts, and generate "
                "text. I understand complex, multi-part sentences and remember what we talked about.",
                "My abilities include question answering, dictionary lookups with definitions and "
                "synonyms, live web search, spelling corrections, and conversation.",
            ],
            'howareyou': [
                "I'm doing great, thank you for asking! All my tensors are in a good mood today. "
                "How can I help you?",
                "Running smoothly! As an AI I don't have feelings exactly, but I'm fully "
                "operational and happy to chat.",
            ],
            'thanks': [
                "You're welcome! Feel free to ask me anything else.",
                "Happy to help! Let me know if you have more questions.",
            ],
            'bye': [
                "Goodbye! It was great chatting with you.",
                "See you later! Come back anytime.",
            ],
            'science': {
                'physics': [
                    "Physics studies the fundamental laws of nature including forces, energy, and matter. "
                    "Key concepts include gravity, electromagnetism, quantum mechanics, and relativity.",
                    "Gravity is a fundamental force of nature that attracts objects with mass toward "
                    "each other. Isaac Newton described gravity mathematically, and Albert Einstein "
                    "later refined our understanding with general relativity.",
                ],
                'biology': [
                    "Biology is the study of living organisms and their interactions. "
                    "Key topics include genetics, evolution, ecology, and cell biology.",
                ],
                'chemistry': [
                    "Chemistry studies the composition, structure, and properties of matter. "
                    "The periodic table organizes elements by atomic number and chemical properties.",
                ],
            },
            'math': [
                "Mathematics provides the language for describing patterns and relationships in nature. "
                "Key branches include algebra, geometry, calculus, statistics, and number theory.",
            ],
            'technology': [
                "Technology encompasses tools and methods created to solve problems. "
                "Computing, AI, the internet, and biotechnology are transforming our world.",
            ],
            'fallback': [
                "That's an interesting topic. Let me share what I know about it.",
                "I'll do my best to help with that. Let me think about it.",
                "That's a great question. Here's my understanding of the topic.",
            ],
        }
        return kb

    def _iter_kb_documents(self):
        """Yield (doc_id, text) pairs for every knowledge base entry."""
        for key, value in self.knowledge_base.items():
            if isinstance(value, list):
                for i, text in enumerate(value):
                    yield f'kb:{key}:{i}', text
            elif isinstance(value, dict):
                for subkey, subtexts in value.items():
                    for i, text in enumerate(subtexts):
                        yield f'kb:{key}:{subkey}:{i}', text

    def _ensure_index(self) -> BM25Index:
        """Build (once) the BM25 index over the knowledge base + corpus."""
        if self._index is None:
            index = BM25Index()
            for doc_id, text in self._iter_kb_documents():
                index.add(doc_id, text)
            for i, text in enumerate(TrainingData.get_training_corpus()):
                index.add(f'corpus:{i}', text)
            self._index = index
        return self._index

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def initialize(self):
        """Initialize or load the AI model."""
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.json')
        model_path = os.path.join(self.model_dir, 'best_model.npz')

        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, 'final_model.npz')

        if os.path.exists(tokenizer_path) and os.path.exists(model_path):
            print("Loading trained model...")
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(tokenizer_path)

            # Load config from model
            loaded = np.load(model_path, allow_pickle=True)
            if 'config' in loaded:
                import json
                config = json.loads(str(loaded['config']))
            else:
                config = {
                    'vocab_size': self.tokenizer.vocab_size,
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 4,
                    'd_ff': 512,
                    'max_seq_len': 256,
                }

            self.model = TransformerLM(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                d_ff=config['d_ff'],
                max_seq_len=config['max_seq_len'],
            )
            self.model.load(model_path)
            self.model.eval_mode()
            self.loaded = True
            print("Model loaded successfully!")
        else:
            print("No trained model found. Using knowledge base, dictionary and search.")
            print("Run 'python train.py' first to train the neural network.")
            self.loaded = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def think(self, user_input: str) -> str:
        """
        Process user input and generate a response.

        The input is fully parsed (clauses, negation, coreference, spelling),
        each clause is answered by the best available source (small talk,
        dictionary, knowledge base, web search, neural model), and the
        per-clause answers are combined into one response.
        """
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        # 1. Understand
        analysis = self.analyzer.analyze(user_input, last_topic=self.last_topic)

        # Note spelling assumptions up front
        spelling_note = ''
        if analysis.spelling_fixes:
            fixes = ', '.join(f"'{w}' -> '{fix}'"
                              for w, fix in list(analysis.spelling_fixes.items())[:3])
            spelling_note = f"(I assumed you meant: {fixes}) "

        # 2. Track the conversation topic for coreference in later turns
        self._update_topic(analysis)

        # 3. Answer each clause
        clause_answers: List[str] = []
        for clause in analysis.clauses[:4]:
            answer = self._answer_clause(clause, analysis)
            if answer:
                clause_answers.append(answer)

        if not clause_answers:
            clause_answers = [self._fallback_answer(analysis)]

        response = self._join_answers(clause_answers)

        # 4. Blend in neural generation when it can add something
        if self.loaded and clause_answers:
            response = self._maybe_blend_neural(response, analysis)

        response = self._post_process(spelling_note + response)

        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        return response

    # ------------------------------------------------------------------
    # Clause routing
    # ------------------------------------------------------------------

    def _answer_clause(self, clause: Clause, analysis: Analysis) -> str:
        """Produce the best answer for a single clause."""
        text = clause.text

        # -- Small talk & identity --------------------------------------
        small = self._match_small_talk(clause)
        if small:
            return small

        # -- Explicit search requests ------------------------------------
        if clause.question_type == 'search' and clause.search_query:
            answer = self._answer_search(clause.search_query)
            if answer:
                return answer
            return self._honest_unknown(clause)

        # -- Dictionary intents ------------------------------------------
        if clause.question_type == 'definition':
            answer = self._answer_definition(clause, analysis)
            if answer:
                return answer

        if clause.question_type == 'spell':
            return self._answer_spell(clause)

        if clause.question_type == 'isword':
            return self._answer_isword(clause)

        if clause.question_type == 'synonyms':
            return self._answer_synonyms(clause)

        # -- Yes/no questions --------------------------------------------
        if clause.question_type == 'yesno':
            answer = self._answer_yesno(clause)
            if answer:
                return answer

        # -- Knowledge retrieval -----------------------------------------
        doc_id, score, coverage, knowledge, confident = self._retrieve_knowledge(clause)
        if knowledge and confident:
            return knowledge

        # -- Web search fallback (only for question-like clauses) ----------
        question_like = (clause.is_question or clause.question_type
                         or clause.entities)
        if self.search.enabled and question_like:
            answer = self._answer_search(clause.text)
            if answer:
                return answer

        # -- Weak local knowledge, but only when it genuinely covers the
        #    question (weighted coverage >= 0.5) - otherwise an honest
        #    "I don't know" beats a coincidental keyword match. --------------
        usable_weak = knowledge and score >= KNOWLEDGE_WEAK_THRESHOLD and \
            coverage >= 0.5
        if usable_weak:
            return knowledge
        if clause.question_type:
            return self._honest_unknown(clause)
        return self._fallback_answer(analysis, clause)

    def _match_small_talk(self, clause: Clause) -> str:
        """Detect greetings, identity, thanks, farewells, capability questions."""
        lemmas = set(clause.focus)
        text = clause.text.lower().strip(' ?!.')
        tokens = set(t.lower() for t in tokenize(clause.text))

        def only(words) -> bool:
            """True if the clause consists (almost) only of these words."""
            content = tokens - words - {'you', 'your', 'me', 'i', 'am', 'are', 'is',
                                        'the', 'a', 'an', 'to', 'so', 'very', 'much',
                                        'today', 'there', 'doing', 'do', 'doing'}
            return len(content) == 0

        greeting_words = {'hi', 'hello', 'hey', 'howdy', 'yo', 'greetings',
                          'morning', 'afternoon', 'evening', 'sup', 'whats', 'up'}
        if only(greeting_words) and (tokens & greeting_words):
            return str(np.random.choice(self.knowledge_base['greetings']))

        if re.search(r"how (are|is) (you|it going|things|your day)", text):
            return str(np.random.choice(self.knowledge_base['howareyou']))

        if lemmas & {'thank', 'thanks', 'thx', 'thankx'} or text in ('thanks', 'thank you'):
            return str(np.random.choice(self.knowledge_base['thanks']))

        if only({'bye', 'goodbye', 'later', 'farewell', 'see', 'soon', 'cya', 'peace'}) \
                and (tokens & {'bye', 'goodbye', 'later', 'farewell', 'cya'}):
            return str(np.random.choice(self.knowledge_base['bye']))

        identity_patterns = [
            r'who are you', r'what are you', r'your name', r'about yourself',
            r'introduce yourself', r'tell me about you\b', r'who made you',
            r'who created you', r'who built you',
        ]
        if any(re.search(p, text) for p in identity_patterns):
            return str(np.random.choice(self.knowledge_base['identity']))

        capability_patterns = [
            r'what can you do', r'your capabilities', r'what do you know',
            r'can you help', r'what are your (abilities|skills|features)',
            r'how can you help', r'what are you capable of',
        ]
        if any(re.search(p, text) for p in capability_patterns):
            return str(np.random.choice(self.knowledge_base['capabilities']))

        return ''

    # ------------------------------------------------------------------
    # Dictionary answers
    # ------------------------------------------------------------------

    def _answer_definition(self, clause: Clause, analysis: Analysis) -> str:
        """Answer 'what does X mean' style questions from the dictionary."""
        target = (clause.definition_target or '').strip()
        if not target:
            # Fall back to the most salient content word
            nouns = [w for w in clause.focus if 'n' in self.dictionary.pos_of(w) or w not in clause.focus]
            target = nouns[0] if nouns else (clause.focus[0] if clause.focus else '')
        if not target:
            return ''

        entry = self.dictionary.lookup(target, use_online=None)
        if entry.found and entry.definitions:
            lines = []
            senses = entry.definitions[:3]
            for i, d in enumerate(senses, 1):
                line = f"{d.text}"
                if d.example and i == 1:
                    line += f' For example: "{d.example}"'
                if d.synonyms and i == 1:
                    line += f" [synonyms: {', '.join(d.synonyms[:4])}]"
                if not line.endswith(('.', '!', '?', '"')):
                    line += '.'
                lines.append(line)
            pos = entry.parts_of_speech[0] if entry.parts_of_speech else 'word'
            word = entry.word[0].upper() + entry.word[1:] if entry.word else entry.word
            return f"{word} ({pos}): " + ' '.join(lines)

        if entry.suggestions:
            return (f"I couldn't find '{target}' in my dictionary. "
                    f"Did you mean: {', '.join(entry.suggestions)}?")

        # Not a dictionary entry - maybe a concept or phrase the local
        # knowledge base literally discusses.
        phrase_match = self._find_phrase_match(target)
        if phrase_match:
            return phrase_match

        # Otherwise search the web for it.
        if self.search.enabled:
            query = f'meaning of {target}' if ' ' not in target else target
            answer = self._answer_search(query)
            if answer:
                return answer
            if self.search.last_status == 'offline':
                return self._honest_unknown(clause)
        if ' ' in target:
            return self._honest_unknown(clause)
        return ''

    def _find_phrase_match(self, phrase: str) -> str:
        """Find a knowledge document that literally discusses `phrase`."""
        needle = ' '.join(phrase.lower().split())
        if len(needle) < 3:
            return ''
        corpus = TrainingData.get_training_corpus()
        for text in corpus:
            if needle in text.lower():
                return self._clean_corpus_answer(text)
        for _doc_id, text in self._iter_kb_documents():
            if needle in text.lower():
                return text
        return ''

    def _answer_spell(self, clause: Clause) -> str:
        """Answer 'how do you spell X' questions."""
        target = (clause.quote or '').strip()
        if not target:
            m = re.search(r'spell(?:ing of| out)?\s+(.+?)\s*\??\s*$', clause.text, re.I)
            if m:
                target = m.group(1).strip(' ?."\'')
        if not target:
            return "Which word would you like me to spell?"

        words = target.split()
        if len(words) == 1:
            word = words[0].lower().strip('.,!?')
            if self.dictionary.is_word(word):
                spelled = '-'.join(word)
                return f"'{word}' is spelled: {spelled} ({len(word)} letters)."
            suggestions = self.dictionary.suggest(word)
            if suggestions:
                return (f"'{word}' is not in the English dictionary. "
                        f"Did you mean: {', '.join(suggestions)}?")
            return f"I can't find '{word}' in my dictionary."
        # Multi-word phrase
        checked = []
        for word in words:
            w = word.lower().strip('.,!?')
            checked.append(w if self.dictionary.is_word(w) else f'{w}(?)')
        return 'That spells out as: ' + ' '.join(checked) + '.'

    def _answer_isword(self, clause: Clause) -> str:
        """Answer 'is X a real word' questions."""
        target = (clause.definition_target or clause.quote or '').strip()
        if not target:
            for pattern in IS_WORD_PATTERNS:
                m = pattern.search(clause.text)
                if m and m.group(1):
                    target = m.group(1).strip(' ?."\'')
                    break
        if not target:
            return ''
        word = target.lower().strip('.,!?"\'')
        if ' ' in word:
            parts_ok = [self.dictionary.is_word(p) for p in word.split()]
            if all(parts_ok):
                return f"'{target}' is a phrase, and every word in it is in the English dictionary."
            return f"'{target}' contains a word I don't recognize."
        if self.dictionary.is_word(word):
            entry = self.dictionary.lookup(word, use_online=False)
            extra = f" ({entry.parts_of_speech[0]})" if entry.parts_of_speech else ''
            return f"Yes, '{word}' is a valid English word{extra}."
        suggestions = self.dictionary.suggest(word)
        if suggestions:
            return (f"'{word}' is not in the English dictionary. "
                    f"Closest words: {', '.join(suggestions)}.")
        return f"No, '{word}' is not in the English dictionary."

    def _answer_synonyms(self, clause: Clause) -> str:
        """Answer 'another word for X' style questions."""
        target = (clause.definition_target or clause.quote or '').strip()
        if not target:
            m = re.search(r'(?:for|of|as)\s+(.+?)\s*\??\s*$', clause.text, re.I)
            if m:
                target = m.group(1).strip(' ?."\'')
        if not target:
            return "Which word do you want synonyms for?"

        synonyms = self.dictionary.synonyms(target)
        if synonyms:
            seen = []
            for s in synonyms:
                if s not in seen:
                    seen.append(s)
            return f"Synonyms for '{target}': {', '.join(seen[:10])}."
        entry = self.dictionary.lookup(target, use_online=False)
        if entry.found:
            return f"I know '{target}', but I don't have synonyms recorded for it."
        return f"I couldn't find '{target}' in my dictionary."

    # ------------------------------------------------------------------
    # Yes/no questions
    # ------------------------------------------------------------------

    def _answer_yesno(self, clause: Clause) -> str:
        """Answer yes/no questions using knowledge, search or the dictionary."""
        doc_id, score, coverage, knowledge, confident = self._retrieve_knowledge(clause)
        if knowledge and confident:
            return knowledge
        if self.search.enabled:
            answer = self._answer_search(clause.text)
            if answer:
                return answer
        if knowledge and score >= KNOWLEDGE_WEAK_THRESHOLD:
            return knowledge
        return ''

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def _answer_search(self, query: str) -> str:
        """Search the web and format the best result as an answer."""
        query = (query or '').strip()
        if not query:
            return ''
        result = self.search.quick_answer(query)
        if result is not None:
            return self._format_search_result(result)
        if self.search.last_status == 'offline':
            return ("I couldn't reach the web right now, so here's my best local "
                    "answer. " + self._fallback_answer(None, None))
        # 'no_results' or unknown: fall through to local knowledge
        return ''

    def _honest_unknown(self, clause: Clause) -> str:
        """Honest answer when nothing local matches a factual question."""
        if self.search.last_status == 'no_results':
            return ("I searched for that but couldn't find a good answer. "
                    "Could you rephrase it or give me a bit more context?")
        if not self.search.enabled:
            return ("I don't have that in my local knowledge and web search is "
                    "currently disabled. Enable it with '/search on' and I'll "
                    "look it up online.")
        return ("I don't have reliable local knowledge about that, and I "
                "couldn't reach the web just now - please try again in a "
                "moment.")

    def _format_search_result(self, result: SearchResult, max_sentences: int = 3) -> str:
        text = result.display.strip()
        sentences = split_sentences(text)
        if len(sentences) > max_sentences:
            text = ' '.join(sentences[:max_sentences])
        if len(text) > 480:
            cut = text[:480]
            last_space = cut.rfind(' ')
            text = cut[:last_space if last_space > 300 else 480] + '...'
        source = result.source.capitalize()
        return f"🌐 {source} — {result.title} ({result.url}): {text}"

    # ------------------------------------------------------------------
    # Knowledge retrieval
    # ------------------------------------------------------------------

    def _retrieve_knowledge(self, clause: Optional[Clause]):
        """
        Rank the knowledge base + corpus with BM25 for a clause.
        Returns (doc_id, score, coverage, text, confident). A match only
        counts as confident when the BM25 score is high AND a good fraction
        of the query terms actually matched, which avoids answering "who won
        the 2022 world cup" with a document that merely contains "world".
        """
        if clause is None:
            return '', 0.0, 0.0, '', False
        terms = clause.focus + [e.lower() for e in clause.entities]
        return self._retrieve_for_terms(terms)

    def _retrieve_for_terms(self, terms: List[str]):
        """BM25 retrieval for an explicit list of query terms."""
        index = self._ensure_index()
        terms = [t for t in terms if t]
        if not terms:
            return '', 0.0, 0.0, '', False
        query = ' '.join(terms)
        # Rare, topical words (photosynthesis) matter more than common ones
        # (work, get), so weight the query accordingly.
        weights = {term: self._term_weight(term) for term in set(terms)}
        results = index.search(query, n=3, term_weights=weights)
        if not results:
            return '', 0.0, 0.0, '', False
        doc_id, score, coverage = results[0]
        self._last_knowledge = [(d, s) for d, s, _ in results]

        confident = (score >= KNOWLEDGE_CONFIDENCE_THRESHOLD and coverage >= 0.5) \
            or score >= KNOWLEDGE_STRONG_SCORE

        text = self._doc_text(doc_id)
        if doc_id.startswith('corpus:'):
            # Corpus Q&A entries have "Question: ... Answer: ..." or
            # "User: ... AI: ..." wrappers; answer with the response part.
            text = self._clean_corpus_answer(text)
            # Augmentation creates single-sentence docs; answer with the
            # richer original text when we can find it.
            text = self._clean_corpus_answer(self._expand_corpus_text(text))
        return doc_id, score, coverage, text, confident

    def _clean_corpus_answer(self, text: str) -> str:
        """Strip Q&A / chat wrappers from a corpus text."""
        if 'Answer:' in text:
            text = text.split('Answer:', 1)[1].strip()
        elif 'AI:' in text:
            idx = text.find('AI:')
            if idx >= 0:
                text = text[idx + 3:].strip()
        return text.strip()

    def _expand_corpus_text(self, text: str) -> str:
        """If `text` is a sentence from a longer corpus entry, return that entry."""
        if len(text) >= 200:
            return text
        corpus = TrainingData.get_training_corpus()
        for candidate in corpus:
            if text != candidate and len(candidate) > len(text) and text in candidate:
                return candidate
        return text

    def _term_weight(self, word: str) -> float:
        """Weight a query term by how informative it is in general English."""
        freq = self.dictionary.frequency(word)
        if freq <= 0:
            return 2.0      # not in the frequency list: likely a rare/proper term
        if freq > 5_000_000:
            return 0.7      # very common word
        if freq > 500_000:
            return 1.0
        if freq > 50_000:
            return 1.5
        return 2.5          # rare word - highly topical

    def _doc_text(self, doc_id: str) -> str:
        """Fetch the stored text for a knowledge index document."""
        if doc_id.startswith('kb:'):
            parts = doc_id.split(':')
            value = self.knowledge_base
            try:
                for key in parts[1:-1]:
                    value = value[key]
                return value[int(parts[-1])]
            except (KeyError, IndexError, ValueError, TypeError):
                return ''
        elif doc_id.startswith('corpus:'):
            corpus = TrainingData.get_training_corpus()
            try:
                return corpus[int(doc_id.split(':')[1])]
            except (IndexError, ValueError):
                return ''
        return ''

    def _find_relevant_knowledge(self, query: str) -> str:
        """
        Find relevant knowledge for a raw query string (compatibility API).
        """
        clause = self.analyzer._analyze_clause(query)
        if clause is None:
            return str(np.random.choice(self.knowledge_base['fallback']))
        _doc, score, _cov, text, _confident = self._retrieve_knowledge(clause)
        if text and score >= KNOWLEDGE_WEAK_THRESHOLD:
            return text
        return str(np.random.choice(self.knowledge_base['fallback']))

    # ------------------------------------------------------------------
    # Topic tracking / coreference
    # ------------------------------------------------------------------

    def _update_topic(self, analysis: Analysis) -> None:
        """Remember the subject of the conversation for coreference."""
        # Prefer definition targets and proper-noun entities
        for clause in analysis.clauses:
            if clause.definition_target:
                self.last_topic = clause.definition_target.lower()
                return
        for clause in analysis.clauses:
            if clause.entities and clause.question_type:
                self.last_topic = ' '.join(e.lower() for e in clause.entities[:3])
                return
        # Otherwise use the most informative content words of the last clause
        for clause in reversed(analysis.clauses):
            if clause.focus:
                self.last_topic = ' '.join(clause.focus[:2])
                return

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    def _join_answers(self, answers: List[str]) -> str:
        """Combine per-clause answers into one coherent reply."""
        answers = [a.strip() for a in answers if a and a.strip()]
        if not answers:
            return str(np.random.choice(self.knowledge_base['fallback']))
        if len(answers) == 1:
            return answers[0]
        if len(answers) == 2:
            return answers[0] + ' ' + answers[1]
        return ' '.join(answers)

    def _fallback_answer(self, analysis: Optional[Analysis],
                         clause: Optional[Clause] = None) -> str:
        if clause is not None and clause.negated:
            return ("Understood - I've noted that. Would you like to talk about "
                    "something else, or ask me to define or search for something?")
        return str(np.random.choice(self.knowledge_base['fallback']))

    def _maybe_blend_neural(self, response: str, analysis: Analysis) -> str:
        """Blend neural generation into the response when it adds information."""
        # Don't pollute clean web-search answers with toy-model text
        if response.startswith('🌐'):
            return response
        try:
            context = ''
            if len(self.conversation_history) > 1:
                recent = self.conversation_history[-3:]
                for msg in recent[:-1]:
                    if msg['role'] == 'user':
                        context += f"User: {msg['content']} "
                    else:
                        context += f"AI: {msg['content']} "
            prompt = f"{context}User: {analysis.resolved_text or analysis.cleaned} AI:"
            neural = self._generate_with_model(prompt, max_tokens=60)
            if neural and len(neural) > 20:
                low = neural.lower()
                # Reject the model parroting the chat format back at us
                if any(m in low for m in ('user:', 'ai:', 'question:', 'answer:')):
                    return response
                # Require some topical overlap with what was asked
                focus = {w for c in analysis.clauses for w in c.focus}
                neural_words = set(re.findall(r'[a-z]+', low))
                if focus and not (focus & neural_words):
                    return response
                return self._blend_responses(response, neural, analysis.cleaned)
        except Exception:
            pass
        return response

    def _generate_with_model(self, prompt: str, max_tokens: int = 80) -> str:
        """Generate text using the neural network model."""
        if not self.loaded or self.model is None:
            return ""

        try:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) == 0:
                return ""

            input_ids = np.array(tokens).reshape(1, -1)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9
            )
            generated_text = self.tokenizer.decode(output_ids.tolist())

            # Clean up
            generated_text = generated_text.strip()
            # Remove the prompt from the beginning if repeated
            if generated_text.lower().startswith(prompt.lower()):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text
        except Exception:
            return ""

    def _blend_responses(self, kb_response: str, neural_response: str, query: str) -> str:
        """Intelligently blend knowledge-based and neural responses."""
        kb_words = set(kb_response.lower().split())
        neural_words = set(neural_response.lower().split())
        new_words = neural_words - kb_words

        if len(new_words) > 5 and len(neural_response) > 30:
            neural_clean = neural_response.split('.')[0] + '.'
            if len(neural_clean) > 20:
                return f"{kb_response} {neural_clean}"

        return kb_response

    _ABBR_PERIODS = ('e.g', 'i.e', 'etc', 'vs', 'Mr', 'Mrs', 'Ms', 'Dr', 'Prof',
                     'approx', 'Inc', 'Ltd', 'Jr', 'Sr')

    def _post_process(self, response: str) -> str:
        """Clean up and improve response quality."""
        response = ' '.join(response.split())

        # Ensure proper ending. Trim an incomplete trailing sentence (e.g.
        # from neural generation), but only when the leftover tail is long
        # enough to be a real incomplete sentence, and only at real
        # sentence-final punctuation (not the periods inside 'e.g.').
        if response and response[-1] not in '.!?\'")':
            cut = -1
            for i in range(len(response) - 1, 0, -1):
                if response[i] in '.!?':
                    prev3 = response[max(0, i - 3):i]
                    if any(prev3.endswith(a) for a in self._ABBR_PERIODS):
                        continue
                    if i == len(response) - 1 or response[i + 1] == ' ':
                        cut = i
                        break
            tail_len = len(response) - cut - 1 if cut > 0 else len(response)
            if cut > len(response) * 0.3 and tail_len > 40:
                response = response[:cut + 1].rstrip()
            else:
                response += '.'

        # Capitalize first letter (leave emoji-prefixed answers alone)
        if response and response[0].isalpha():
            response = response[0].upper() + response[1:]

        return response

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.last_topic = ''
        print("Conversation history cleared.")

    def set_search_enabled(self, enabled: bool) -> None:
        self.search.enabled = enabled

    @property
    def search_enabled(self) -> bool:
        return self.search.enabled
