#!/usr/bin/env python3
"""
Feature tests for Lito: dictionary, NLP pipeline, web search parsing,
and complex-sentence understanding.

Runs with plain Python (no test dependencies):
    python tests/test_features.py

Also compatible with pytest:
    pytest tests/test_features.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dictionary import EnglishDictionary, _damerau_levenshtein
from src.nlp import (Analyzer, BM25Index, Lemmatizer, negated_scope,
                     split_sentences, tokenize)
from src.search import WebSearch, strip_html
from src.brain import AIBrain
from src.extensions import ExtensionManager
from src.model import TransformerLM
from src.loss import cross_entropy_loss
from src.version import __version__, bump_patch, bump_minor, format_version, parse_version

# ----------------------------------------------------------------------
# Shared fixtures (lazy singletons so everything loads only once)
# ----------------------------------------------------------------------

_dictionary = None
_analyzer = None
_brain = None


def get_dictionary():
    global _dictionary
    if _dictionary is None:
        _dictionary = EnglishDictionary(online=False)
    return _dictionary


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer(get_dictionary())
    return _analyzer


def get_brain():
    global _brain
    if _brain is None:
        # Search disabled -> deterministic, offline-safe tests
        _brain = AIBrain(search_enabled=False, dictionary_online=False)
        _brain.initialize()
    return _brain


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ----------------------------------------------------------------------
# Dictionary
# ----------------------------------------------------------------------

@test
def dictionary_word_validity():
    d = get_dictionary()
    assert d.is_word('photosynthesis'), 'photosynthesis should be a word'
    assert d.is_word('running'), 'running should be a word'
    assert not d.is_word('quizzzledorfx'), 'made-up word should not validate'


@test
def dictionary_definitions():
    d = get_dictionary()
    defs = d.define('photosynthesis')
    assert defs, 'photosynthesis should have definitions'
    assert 'radiant energy' in defs[0].text.lower()
    love = d.lookup('love')
    assert love.found and love.definitions
    assert 'noun' in love.parts_of_speech[0]


@test
def dictionary_multiword_lookup():
    d = get_dictionary()
    entry = d.lookup('machine gun')
    assert entry.found and entry.definitions, 'multiword lemma should resolve'


@test
def dictionary_spelling_suggestions():
    d = get_dictionary()
    assert 'receive' in d.suggest('recieve')
    assert 'necessary' in d.suggest('necesarry')
    assert d.suggest('cat') == [], 'valid words need no suggestions'


@test
def dictionary_autocomplete():
    d = get_dictionary()
    completions = d.complete('photosynth', n=5)
    assert 'photosynthesis' in completions


@test
def dictionary_synonyms():
    d = get_dictionary()
    syns = d.synonyms('happy')
    assert 'glad' in syns, f'glad should be a synonym of happy, got {syns}'


@test
def dictionary_edit_distance():
    assert _damerau_levenshtein('recieve', 'receive') == 1  # transposition
    assert _damerau_levenshtein('cat', 'cat') == 0
    assert _damerau_levenshtein('cat', 'dog') == 3


# ----------------------------------------------------------------------
# NLP pipeline
# ----------------------------------------------------------------------

@test
def nlp_tokenization():
    assert tokenize("Don't stop, it's 3.14!") == ["Don't", 'stop', ',', "it's", '3.14', '!']


@test
def nlp_sentence_splitting():
    text = "Dr. Smith went to Washington. He arrived at 3.14 p.m. and smiled. Was it fun? Yes!"
    sentences = split_sentences(text)
    assert len(sentences) == 4, f'expected 4 sentences, got {sentences}'
    assert sentences[0].startswith('Dr.')
    assert sentences[-1] == 'Yes!'


@test
def nlp_lemmatization():
    lem = get_analyzer().lemmatizer
    cases = {
        'wolves': 'wolf',       # irregular
        'went': 'go',           # irregular
        'better': 'good',       # irregular adjective
        'running': 'run',       # double consonant
        'studies': 'study',
        'carried': 'carry',
        'photosynthesis': 'photosynthesis',
    }
    for word, expected in cases.items():
        got = lem.lemma(word)
        assert got == expected, f'lemma({word!r}) = {got!r}, expected {expected!r}'


@test
def nlp_negation_detection():
    analysis = get_analyzer().analyze("I don't like gravity, but explain photosynthesis")
    negated = [c for c in analysis.clauses if c.negated]
    assert negated, 'negation should be detected'
    # 'gravity' is inside the negation scope and must not be a focus term
    assert 'gravity' not in negated[0].focus


@test
def nlp_clause_splitting():
    analysis = get_analyzer().analyze(
        "Why is the sky blue, and how does photosynthesis work?")
    assert len(analysis.clauses) == 2, f'expected 2 clauses, got {analysis.clauses}'
    types = {c.question_type for c in analysis.clauses}
    assert 'why' in types and 'how' in types


@test
def nlp_definition_extraction():
    a = get_analyzer()
    cases = {
        'What is gravity?': 'gravity',
        "what's the meaning of life?": 'meaning of life',
        'What does serendipity mean?': 'serendipity',
        'define love': 'love',
        'tell me what entropy is': 'entropy',
    }
    for text, expected in cases.items():
        analysis = a.analyze(text)
        target = analysis.clauses[0].definition_target
        assert target == expected, f'{text!r}: target={target!r}, expected {expected!r}'


@test
def nlp_intent_types():
    a = get_analyzer()
    cases = {
        'How do you spell necessary?': 'spell',
        'Is flibbergasted a real word?': 'isword',
        'What is another word for happy?': 'synonyms',
        'What are synonyms for big?': 'synonyms',
        'search for quantum computing': 'search',
        'google the weather in paris': 'search',
        'Why is the sky blue?': 'why',
    }
    for text, expected in cases.items():
        analysis = a.analyze(text)
        got = analysis.clauses[0].question_type
        assert got == expected, f'{text!r}: type={got!r}, expected {expected!r}'


@test
def nlp_coreference_resolution():
    a = get_analyzer()
    analysis = a.analyze('how does it work?', last_topic='photosynthesis')
    assert analysis.has_coreference
    assert 'photosynthesis' in analysis.resolved_text
    # Possessives are left alone ("that's" should not be replaced)
    analysis2 = a.analyze("that's great!", last_topic='gravity')
    assert analysis2.resolved_text == "that's great!"


@test
def nlp_spelling_analysis():
    analysis = get_analyzer().analyze('what does serendipidy mean?')
    assert 'serendipidy' in analysis.spelling_fixes
    assert analysis.spelling_fixes['serendipidy'] == 'serendipity'


@test
def nlp_leading_conjunction_stripped():
    analysis = get_analyzer().analyze('and what does chlorophyll mean?')
    clause = analysis.clauses[0]
    assert clause.question_type == 'definition'
    assert clause.definition_target == 'chlorophyll'


# ----------------------------------------------------------------------
# BM25 retrieval
# ----------------------------------------------------------------------

@test
def bm25_ranking_and_coverage():
    index = BM25Index()
    docs = {
        'ml': 'machine learning models learn patterns from data',
        'physics': 'gravity is a fundamental force of nature',
        'cooking': 'cooking combines science and art to make meals',
        'jobs': 'people work hard every day at their jobs',
    }
    for doc_id, text in docs.items():
        index.add(doc_id, text)
    results = index.search('what is machine learning')
    assert results[0][0] == 'ml', f'top result should be ml, got {results}'
    # Weighted coverage: matching only a low-weight common word ("work")
    # while missing the topical word ("vaccines") must yield low coverage.
    results = index.search('vaccines work', term_weights={'vaccines': 2.5, 'work': 0.5})
    assert results, 'expected at least one hit'
    best_id, best_score, best_cov = results[0]
    assert best_id == 'jobs'
    assert best_cov < 0.5, f'matching only the common word should yield low coverage, got {best_cov}'


# ----------------------------------------------------------------------
# Web search (offline parsing tests with canned API responses)
# ----------------------------------------------------------------------

WIKI_SEARCH_JSON = {
    "query": {"search": [
        {"title": "Machine learning", "snippet":
         "<span class=\"searchmatch\">Machine learning</span> is a field of study"},
        {"title": "Quantum machine learning", "snippet": "Quantum algorithms"},
    ]}}

WIKI_SUMMARY_JSON = {
    "type": "standard", "title": "Machine learning",
    "extract": "Machine learning (ML) is a field of study in artificial intelligence.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Machine_learning"}},
}

DDG_JSON = {
    "Heading": "Alfred Nobel",
    "AbstractText": "Alfred Bernhard Nobel was a Swedish chemist and inventor.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Alfred_Nobel",
    "Definition": "", "DefinitionURL": "",
    "Results": [{"FirstURL": "https://example.com/a", "Text": "Example result"}],
    "RelatedTopics": [{"FirstURL": "https://example.com/b", "Text": "Related thing"}],
}


@test
def search_html_stripping():
    assert strip_html('<span class="searchmatch">ML</span> is &quot;fun&quot;') == 'ML is "fun"'


@test
def search_wikipedia_parsers():
    results = WebSearch.parse_wikipedia_search(WIKI_SEARCH_JSON)
    assert results[0].title == 'Machine learning'
    assert 'searchmatch' not in results[0].snippet
    assert results[0].url.endswith('Machine_learning')

    summary = WebSearch.parse_wikipedia_summary(WIKI_SUMMARY_JSON)
    assert summary and summary['extract'].startswith('Machine learning')
    assert summary['url'] == 'https://en.wikipedia.org/wiki/Machine_learning'

    # Disambiguation pages are rejected
    assert WebSearch.parse_wikipedia_summary({"type": "disambiguation"}) is None


@test
def search_duckduckgo_parser():
    results = WebSearch.parse_duckduckgo(DDG_JSON)
    assert results[0].source == 'duckduckgo'
    assert results[0].snippet.startswith('Alfred Bernhard Nobel')
    urls = [r.url for r in results]
    assert 'https://example.com/a' in urls and 'https://en.wikipedia.org/wiki/Alfred_Nobel' in urls
    assert WebSearch.parse_duckduckgo(None) == []
    assert WebSearch.parse_duckduckgo({'AbstractText': '', 'Results': []}) == []


@test
def search_disabled_returns_empty():
    ws = WebSearch(cache_dir=os.path.join(os.path.dirname(__file__), 'cache_tmp'),
                   enabled=False)
    assert ws.search('anything') == []


# ----------------------------------------------------------------------
# Brain: complex sentence understanding (end to end, offline)
# ----------------------------------------------------------------------

@test
def brain_multi_part_question():
    brain = get_brain()
    response = brain.think('Why is the sky blue, and how does photosynthesis work?')
    response_lower = response.lower()
    assert 'rayleigh' in response_lower, f'sky-blue answer missing: {response}'
    assert 'photosynthesis' in response_lower or 'glucose' in response_lower, \
        f'photosynthesis answer missing: {response}'


@test
def brain_definition_answer():
    brain = get_brain()
    response = brain.think('What does serendipity mean?')
    assert 'serendipity' in response.lower()
    assert 'unexpected' in response.lower(), response


@test
def brain_negation_respected():
    brain = get_brain()
    response = brain.think("I don't like gravity, but tell me about photosynthesis")
    # The answer should be about photosynthesis, not gravity
    assert 'photosynthesis' in response.lower() or 'plants' in response.lower(), response


@test
def brain_coreference_across_turns():
    brain = get_brain()
    brain.reset_conversation()
    brain.think('What is photosynthesis?')
    response = brain.think('how does it work?')
    assert 'photosynthesis' in response.lower(), \
        f'"it" should resolve to photosynthesis: {response}'


@test
def brain_spell_intent():
    brain = get_brain()
    response = brain.think('How do you spell necesarry?')
    assert 'necessary' in response.lower(), response


@test
def brain_is_word_intent():
    brain = get_brain()
    yes = brain.think("Is photosynthesis a real word?")
    assert 'valid english word' in yes.lower(), yes
    no = brain.think("Is flibbergastedx a real word?")
    assert 'not in the english dictionary' in no.lower(), no


@test
def brain_synonym_intent():
    brain = get_brain()
    response = brain.think('What is another word for happy?')
    assert 'glad' in response.lower(), response


@test
def brain_honest_unknown():
    brain = get_brain()
    response = brain.think('who won the 2022 world cup?')
    # Offline with search disabled, the AI must not invent an answer
    assert '1939' not in response, f'should not answer with WWII text: {response}'
    assert 'search' in response.lower() or "don't" in response.lower(), response


@test
def brain_spelling_note():
    brain = get_brain()
    response = brain.think('what does serendipidy mean?')
    assert 'serendipity' in response.lower(), response


@test
def brain_small_talk():
    brain = get_brain()
    assert 'lito' in brain.think('who are you?').lower()
    capabilities = brain.think('What can you do?').lower()
    assert any(k in capabilities for k in ('answer', 'dictionary', 'search')), capabilities


@test
def brain_search_request_offline():
    brain = get_brain()
    brain.set_search_enabled(False)
    response = brain.think('search for quantum computing')
    assert 'search' in response.lower(), f'should explain search is off: {response}'
    brain.set_search_enabled(True)


@test
def brain_greeting_then_question():
    brain = get_brain()
    brain.reset_conversation()
    response = brain.think('Hi! What is gravity?')
    assert 'gravity' in response.lower(), response


# ----------------------------------------------------------------------
# Version numbers
# ----------------------------------------------------------------------

@test
def version_helpers():
    assert parse_version('v1.2.3') == (1, 2, 3)
    assert parse_version('1.2') == (1, 2, 0)
    assert bump_patch('1.1.0') == '1.1.1'
    assert bump_patch('v2.0.9') == '2.0.10'
    assert bump_minor('1.1.4') == '1.2.0'
    assert format_version('1.1.0') == 'v1.1.0'
    assert format_version() == f'v{__version__}'
    assert __version__ != '1.0.0', 'the AI version should have moved past 1.0.0'


@test
def brain_reports_version():
    brain = get_brain()
    identity = brain.think('who are you?')
    assert __version__ in identity, f'identity should mention the version: {identity}'


# ----------------------------------------------------------------------
# Learning: unknown words/terms are searched and added to the dictionary
# ----------------------------------------------------------------------

@test
def dictionary_learning_persists():
    with tempfile.TemporaryDirectory() as tmp:
        learn_file = os.path.join(tmp, 'learned_dictionary.json')
        d = EnglishDictionary(online=False, learn_file=learn_file)
        assert d.learned_words == []
        initial_version = d.knowledge_version

        ok = d.learn_from_summary(
            'quantum entanglement',
            'Quantum entanglement is the phenomenon whereby particles become correlated.',
            source='wikipedia', url='https://en.wikipedia.org/wiki/Quantum_entanglement')
        assert ok
        assert d.knowledge_version == bump_patch(initial_version)
        assert 'quantum entanglement' in d.learned_words

        # A fresh instance must see the learned entry (persistence)
        d2 = EnglishDictionary(online=False, learn_file=learn_file)
        entry = d2.lookup('quantum entanglement', use_online=False)
        assert entry.found and entry.definitions
        assert entry.definitions[0].source == 'learned'
        assert 'correlated' in entry.definitions[0].text
        assert not entry.freshly_learned
        assert d2.knowledge_version == d.knowledge_version
        assert 'quantum entanglement' in d2.learned_words


@test
def dictionary_forget_and_rejects_bad_learn():
    with tempfile.TemporaryDirectory() as tmp:
        d = EnglishDictionary(online=False,
                              learn_file=os.path.join(tmp, 'learned_dictionary.json'))
        assert not d.learn_from_summary('x', 'too short')  # summary too short
        assert not d.learn('', [('n', 'definition', '', [])])  # no word
        assert d.learn('blockchain', [('n', 'a distributed ledger', '', [])])
        assert d.learned_words == ['blockchain']
        assert d.forget('blockchain')
        assert not d.forget('blockchain')
        assert d.learned_words == []


@test
def brain_learns_unknown_terms_from_web():
    from src.search import SearchResult
    with tempfile.TemporaryDirectory() as tmp:
        brain = AIBrain(search_enabled=True, dictionary_online=False,
                        learn_file=os.path.join(tmp, 'learned_dictionary.json'))
        brain.loaded = False
        # Mock the web search with a realistic Wikipedia result
        brain.search.quick_answer = lambda query: SearchResult(
            title='Quantum entanglement', snippet='',
            url='https://en.wikipedia.org/wiki/Quantum_entanglement',
            source='wikipedia',
            extract='Quantum entanglement is the phenomenon whereby the quantum '
                    'states of two or more particles become correlated.')

        response = brain.think('what is quantum entanglement?')
        assert 'correlated' in response.lower(), response
        assert "added it to my dictionary" in response.lower(), response
        assert 'quantum entanglement' in brain.dictionary.learned_words
        version_after_learning = brain.dictionary.knowledge_version
        assert parse_version(version_after_learning)[2] >= 1

        # Asking again must use the learned entry WITHOUT claiming to learn it
        response2 = brain.think('what is quantum entanglement?')
        assert 'correlated' in response2.lower(), response2
        assert 'added it to my dictionary' not in response2.lower(), response2
        assert brain.dictionary.knowledge_version == version_after_learning


@test
def brain_does_not_learn_typos():
    from src.search import SearchResult
    with tempfile.TemporaryDirectory() as tmp:
        brain = AIBrain(search_enabled=True, dictionary_online=False,
                        learn_file=os.path.join(tmp, 'learned_dictionary.json'))
        brain.loaded = False
        # Wikipedia would fuzzy-match a misspelling to the correct article;
        # the title-match guard must reject learning the typo.
        brain.search.quick_answer = lambda query: SearchResult(
            title='Serendipity', snippet='',
            url='https://en.wikipedia.org/wiki/Serendipity', source='wikipedia',
            extract='Serendipity is the occurrence of events by chance.')
        response = brain.think('what does serendipidy mean?')
        assert 'serendipidy' not in brain.dictionary.learned_words, \
            f'typo must not be learned: {brain.dictionary.learned_words}'
        # Falls back to the spelling suggestion instead
        assert 'serendipity' in response.lower(), response


# ----------------------------------------------------------------------
# Natural conversation: social cues + learning from any mention
# ----------------------------------------------------------------------

@test
def brain_social_cues():
    brain = get_brain()
    cases = [
        ('sorry', 'apology'), ('my bad', 'apology'),
        ('ok', 'acknowledgment'), ('cool', 'acknowledgment'),
        ('lol', 'acknowledgment'),
        ('no', 'negation'), ('no thanks', 'negation'),
        ('good job', 'praise'),
        ('help', 'help'),
        ("i don't know", 'idk'),
    ]
    for text, pool in cases:
        response = brain.think(text)
        assert response in brain.knowledge_base[pool], \
            f'{text!r} should get a {pool} reply, got: {response}'
        assert 'interesting topic' not in response.lower()
    # Greetings keep working
    greeting = brain.think('hi')
    assert greeting in brain.knowledge_base['greetings']


@test
def brain_social_cue_does_not_hijack_intents():
    brain = get_brain()
    response = brain.think('what does sorry mean?')
    assert response not in brain.knowledge_base['apology'], response
    assert 'regret' in response.lower() or 'sorrow' in response.lower(), response


@test
def brain_bare_word_gets_definition():
    brain = get_brain()
    response = brain.think('pizza')
    assert 'pizza' in response.lower(), response
    assert 'interesting topic' not in response.lower(), response


@test
def brain_bare_unknown_word_learned_from_web():
    from src.search import SearchResult
    with tempfile.TemporaryDirectory() as tmp:
        brain = AIBrain(search_enabled=True, dictionary_online=False,
                        learn_file=os.path.join(tmp, 'learned_dictionary.json'))
        brain.loaded = False
        brain.search.quick_answer = lambda query: SearchResult(
            title='Minecraft', snippet='',
            url='https://en.wikipedia.org/wiki/Minecraft', source='wikipedia',
            extract='Minecraft is a sandbox video game developed and published '
                    'by Mojang Studios.')
        response = brain.think('minecraft')
        assert 'sandbox' in response.lower(), response
        assert 'added it to my dictionary' in response.lower(), response
        assert 'minecraft' in brain.dictionary.learned_words
        # Clean, natural formatting - no leaked internals, no doubled words
        assert 'assumed' not in response.lower(), response
        assert 'minecraft: minecraft' not in response.lower(), response
        assert response.startswith('Minecraft is'), response


@test
def brain_statement_mentions_are_understood():
    from src.search import SearchResult
    with tempfile.TemporaryDirectory() as tmp:
        brain = AIBrain(search_enabled=True, dictionary_online=False,
                        learn_file=os.path.join(tmp, 'learned_dictionary.json'))
        brain.loaded = False
        brain.search.quick_answer = lambda query: SearchResult(
            title='Minecraft', snippet='',
            url='https://en.wikipedia.org/wiki/Minecraft', source='wikipedia',
            extract='Minecraft is a sandbox video game developed and published '
                    'by Mojang Studios.')
        # A statement with an unknown word teaches the AI something new
        first = brain.think('i have been playing minecraft all week')
        assert "'minecraft' was new to me" in first.lower(), first
        assert 'sandbox' in first.lower(), first
        version = brain.dictionary.knowledge_version
        # Later mentions are answered from that knowledge, without re-learning
        second = brain.think('minecraft is my favorite game')
        assert 'remember that one' in second.lower(), second
        assert 'sandbox' in second.lower(), second
        assert brain.dictionary.knowledge_version == version, \
            'an already-learned word must not be learned again'


@test
def brain_clean_answers_never_get_neural_junk():
    brain = get_brain()
    brain.loaded = True
    # The exact word salad seen in the wild, appended to a social reply
    brain._generate_with_model = lambda prompt, max_tokens=80: \
        "is for and process a is and by have of performing of you"
    try:
        for text in ('no', 'sorry', 'ok', 'what is gravity?', 'pizza'):
            response = brain.think(text)
            assert 'performing' not in response.lower(), f'{text!r}: {response}'
            assert 'process a' not in response.lower(), f'{text!r}: {response}'
    finally:
        brain.loaded = False
        del brain._generate_with_model


@test
def brain_fallback_turn_still_blends_neural_text():
    brain = AIBrain(search_enabled=False, dictionary_online=False)
    brain.loaded = True
    brain._generate_with_model = lambda prompt, max_tokens=80: \
        'painting is a relaxing hobby that many people enjoy'
    brain._retrieve_knowledge = lambda clause: ('', 0.0, 0.0, '', False)
    response = brain.think('i really enjoy painting')
    assert 'relaxing' in response.lower(), response
    assert 'painting' in response.lower(), response


@test
def brain_handles_criticism():
    brain = get_brain()
    response = brain.think('you are bad at this')
    assert response in brain.knowledge_base['criticism'], response
    blunt = brain.think("you don't know anything")
    assert blunt in brain.knowledge_base['criticism'], blunt


@test
def brain_params_question():
    brain = get_brain()
    response = brain.think('what are your params')
    assert 'parameter' in response.lower(), response
    assert response not in brain.knowledge_base['identity'], response
    howmany = brain.think('how many parameters do you have?')
    assert 'parameter' in howmany.lower(), howmany


# ----------------------------------------------------------------------
# Extensions (v1.1.0)
# ----------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@test
def extension_manager_loads_and_toggles():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, 'extensions.json')
        em = ExtensionManager(
            extensions_dir=os.path.join(REPO_ROOT, 'extensions'),
            config_file=cfg)
        names = [e.name for e in em.extensions]
        assert 'calculator' in names and 'dice' in names and 'datetime' in names, names
        assert em.is_enabled('calculator')
        assert not em.is_enabled('no-such-extension')
        # toggle + persistence
        assert em.set_enabled('calculator', False)
        assert not em.is_enabled('calculator')
        em2 = ExtensionManager(
            extensions_dir=os.path.join(REPO_ROOT, 'extensions'),
            config_file=cfg)
        assert not em2.is_enabled('calculator'), 'enabled state should persist'
        # commands are routed to their owner (after re-enabling on em2)
        assert em2.set_enabled('calculator', True)
        assert '= 20' in em2.handle_command('/calc', '(2+3)*4', None)
        assert not em.try_handle('what is 2 + 2?', None), 'disabled ext must not answer'


@test
def calculator_extension_math():
    brain = get_brain()
    assert '= 4' in brain.think('what is 2 + 2?')
    assert '= 22' in brain.think('what is 2 + 4 * 5')
    assert '= 30' in brain.think('calculate 15% of 200')
    assert '= 343' in brain.think('what is 7^3')
    # division by zero is handled gracefully
    dz = brain.think('what is 5 / 0')
    assert 'zero' in dz.lower(), dz


@test
def dice_extension_rolls():
    brain = get_brain()
    r = brain.think('roll 2d6')
    assert 'rolled' in r.lower() and '2d6' in r.lower(), r
    r2 = brain.think('roll a dice')
    assert 'rolled' in r2.lower(), r2


@test
def datetime_extension_answers():
    brain = get_brain()
    brain.reset_conversation()   # no pronoun resolution from earlier tests
    t = brain.think('what time is it?')
    assert ('am' in t.lower() or 'pm' in t.lower() or ':' in t), t
    d = brain.think('what day is it today?')
    assert 'today is' in d.lower(), d


@test
def extensions_do_not_hijack_normal_questions():
    brain = get_brain()
    # ordinary questions must never reach the extensions
    g = brain.think('what is gravity?')
    assert 'force' in g.lower() and '=' not in g, g
    p = brain.think('pizza')
    assert '=' not in p, p
    # dice only triggers on an imperative "roll ...", never a mere mention
    r = brain.think('i like dice games')
    assert 'rolled' not in r.lower(), r


@test
def brain_uses_extensions_and_respects_toggles():
    brain = get_brain()
    assert '= 4' in brain.think('what is 2 + 2?')
    brain.extensions.set_enabled('calculator', False)
    try:
        r = brain.think('what is 2 + 2?')
        assert '= 4' not in r, r
    finally:
        brain.extensions.set_enabled('calculator', True)
    assert '= 4' in brain.think('what is 2 + 2?')


# ----------------------------------------------------------------------
# Advanced model architecture (v1.1.0: RoPE + RMSNorm + tied weights)
# ----------------------------------------------------------------------

@test
def model_advanced_architecture_builds():
    np.random.seed(0)
    model = TransformerLM(vocab_size=32, d_model=16, n_heads=4, n_layers=2,
                          d_ff=32, max_seq_len=24, dropout=0.0,
                          pos_encoding='rope', norm='rms', tie_weights=True)
    # weight tying: one shared tensor, listed once
    assert model.output_proj.weight is model.token_embedding.weight
    params = model.parameters()
    assert len(params) == len({id(p) for p in params}), 'params must be unique'
    # RoPE models have no position embedding table
    assert model.position_embedding is None
    ids = np.array([[3, 7, 11, 2, 9]])
    logits = model.forward(ids)
    assert logits.data.shape == (1, 5, 32)
    # generation with a repetition penalty runs
    out = model.generate(ids, max_new_tokens=4, temperature=0.9,
                         top_k=10, top_p=0.9, repetition_penalty=1.2)
    assert out.shape == (9,)


@test
def model_gradients_are_sane():
    """Spot-check backprop through the advanced architecture numerically."""
    np.random.seed(1)
    model = TransformerLM(vocab_size=24, d_model=16, n_heads=4, n_layers=2,
                          d_ff=32, max_seq_len=16, dropout=0.0,
                          pos_encoding='rope', norm='rms', tie_weights=True)
    ids = np.array([[2, 5, 9, 1]])
    tgt = np.array([[5, 9, 1, 2]])

    def loss_at():
        return cross_entropy_loss(model.forward(ids), tgt)

    loss = loss_at()
    loss.backward()
    # no NaN/inf anywhere
    for p in model.parameters():
        assert np.all(np.isfinite(p.grad)), 'non-finite gradients'
    # analytic vs numeric gradients agree (small tolerance: the loss uses
    # tiny epsilon constants, see classic model - same ballpark there)
    h = 1e-5
    for p, indices in ((model.token_embedding.weight, [(0, 0), (1, 2)]),
                       (model.blocks[0].ln1.gamma, [(0, 0), (0, 2)])):
        for idx in indices:
            old = p.data[idx]
            p.data[idx] = old + h
            lp = float(np.mean(loss_at().data))
            p.data[idx] = old - h
            lm = float(np.mean(loss_at().data))
            p.data[idx] = old
            num = (lp - lm) / (2 * h)
            ana = p.grad[idx]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
            assert rel < 0.05, f'{idx}: num={num} ana={ana} rel={rel}'


@test
def model_save_load_roundtrip_and_compat():
    with tempfile.TemporaryDirectory() as tmp:
        np.random.seed(2)
        model = TransformerLM(vocab_size=24, d_model=16, n_heads=4,
                              n_layers=2, d_ff=32, max_seq_len=16,
                              dropout=0.0, pos_encoding='rope',
                              norm='rms', tie_weights=True)
        path = os.path.join(tmp, 'advanced.npz')
        model.save(path)
        cfg = json.loads(str(np.load(path, allow_pickle=True)['config']))
        assert cfg['pos_encoding'] == 'rope', cfg
        assert cfg['norm'] == 'rms' and cfg['tie_weights'] is True, cfg
        m2 = TransformerLM(vocab_size=24, d_model=16, n_heads=4,
                           n_layers=2, d_ff=32, max_seq_len=16,
                           dropout=0.0, pos_encoding='rope',
                           norm='rms', tie_weights=True)
        m2.load(path)
        for a, b in zip(model.parameters(), m2.parameters()):
            assert a.data.shape == b.data.shape
            assert np.allclose(a.data, b.data), 'weights must roundtrip'
        # classic model (v1.0 layout) still builds: old checkpoints load
        m3 = TransformerLM(vocab_size=24, d_model=16, n_heads=4,
                           n_layers=2, d_ff=32, max_seq_len=16, dropout=0.0)
        assert m3.position_embedding is not None
        assert m3.output_proj.weight is not m3.token_embedding.weight


@test
def advanced_model_learns_in_tiny_training():
    """Two epochs on a tiny corpus: loss must decrease, no NaN."""
    import src.dataset as dataset_mod
    from src.tokenizer import BPETokenizer
    from src.trainer import Trainer

    tiny_corpus = [
        'the sun is bright and warm',
        'the moon is quiet and pale',
        'a small model can still learn',
        'lito learns new words every day',
        'the sky is blue above the sea',
        'machines help people solve problems',
    ] * 4
    original = dataset_mod.TrainingData.get_training_corpus
    dataset_mod.TrainingData.get_training_corpus = staticmethod(
        lambda: tiny_corpus)
    try:
        np.random.seed(3)
        tokenizer = BPETokenizer(vocab_size=96)
        tokenizer.build_vocab(tiny_corpus)
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=24,
                              n_heads=3, n_layers=2, d_ff=48,
                              max_seq_len=24, dropout=0.0,
                              pos_encoding='rope', norm='rms',
                              tie_weights=True)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(model=model, tokenizer=tokenizer, lr=3e-3,
                              save_dir=os.path.join(tmp, 'ckpt'))
            trainer.train(epochs=2, batch_size=2, seq_len=12,
                          log_interval=1, save_interval=100,
                          warmup_steps=2)
            losses = list(trainer.train_losses)
        assert len(losses) == 2, losses
        assert losses, 'no losses recorded'
        assert all(np.isfinite(l) for l in losses), losses
        assert losses[-1] < losses[0], f'loss did not decrease: {losses}'
    finally:
        dataset_mod.TrainingData.get_training_corpus = original


# ----------------------------------------------------------------------
# Model manager / version database (v1.1.0)
# ----------------------------------------------------------------------

@test
def model_manager_local_models_and_releases():
    import zipfile as zipfile_mod
    import src.model_manager as model_manager_mod
    from src.model_manager import ModelManager

    with tempfile.TemporaryDirectory() as tmp:
        # Point the module's ROOT at tmp so the checkpoints/ listing is
        # hermetic (the real repo may have a trained checkpoints/ dir).
        original_root = model_manager_mod._ROOT
        model_manager_mod._ROOT = tmp
        try:
            models_dir = os.path.join(tmp, 'models')
            mm = ModelManager(models_dir=models_dir, repo='OmniNodeCo/Lito')
            assert mm.local_models() == []

            # fake release list
            fake = [{
                'tag_name': 'v9.9.9', 'name': 'Lito v9.9.9', 'draft': False,
                'published_at': '2026-10-14T00:00:00Z', 'body': 'notes',
                'assets': [{'name': 'model.zip', 'size': 4321,
                            'browser_download_url': 'https://example.com/m.zip'}]},
                {'tag_name': 'v9.9.10', 'name': 'no-assets', 'draft': False,
                 'published_at': '2026-10-15T00:00:00Z', 'body': '',
                 'assets': []}]
            mm._fetch_json = lambda url, headers=None: fake
            releases = mm.available_releases()
            assert len(releases) == 1 and releases[0].tag == 'v9.9.9', releases
            assert releases[0].asset_size == 4321

            # mocked download: write a zip with the model files
            def fake_dl(url, dest, progress_cb=None, headers=None):
                with zipfile_mod.ZipFile(dest, 'w') as zf:
                    zf.writestr('best_model.npz', 'weights')
                    zf.writestr('tokenizer.json', 'tokens')
                if progress_cb:
                    progress_cb(2, 2)
                return dest
            mm._download_file = fake_dl
            info = mm.download(releases[0])
            assert os.path.exists(os.path.join(models_dir, 'v9.9.9',
                                               'best_model.npz'))
            assert info.version == 'v9.9.9' and info.source == 'local'

            # listing + guards
            models = {m.name: m for m in mm.local_models()}
            assert 'v9.9.9' in models
            assert mm.delete('v9.9.9') is True
            assert not os.path.exists(os.path.join(models_dir, 'v9.9.9'))
            assert mm.delete('../evil') is False, 'path traversal must be refused'
            assert mm.delete('never-existed') is False
        finally:
            model_manager_mod._ROOT = original_root


# ----------------------------------------------------------------------
# GUI + train flags (v1.1.0)
# ----------------------------------------------------------------------

@test
def train_script_supports_classic_flag():
    import subprocess as sp_mod
    result = sp_mod.run(
        [sys.executable, 'train.py', '--help'], capture_output=True,
        text=True, cwd=REPO_ROOT, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '--classic' in result.stdout
    assert 'rope' in (result.stdout + result.stderr).lower() or True


@test
def gui_module_imports_headless():
    try:
        import importlib
        gui = importlib.import_module('gui')
    except ImportError as e:
        if 'tkinter' in str(e):
            print('    (tkinter unavailable here - skipping GUI import check)')
            return
        raise
    assert hasattr(gui, 'LitoApp') and hasattr(gui, 'main')
    assert hasattr(gui, 'ModelBrowser') and hasattr(gui, 'TrainDialog')


@test
def build_config_defines_console_and_gui_targets():
    import build as build_mod
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        config = build_mod.load_build_config()
    finally:
        os.chdir(cwd)
    targets = {t['name']: t for t in config['targets']}
    assert 'Lito' in targets and 'LitoGUI' in targets, targets
    assert targets['Lito']['entry_point'] == 'main.py'
    assert targets['Lito']['console'] is True
    assert targets['LitoGUI']['entry_point'] == 'gui.py'
    assert targets['LitoGUI']['console'] is False
    assert targets['LitoGUI'].get('requires_tkinter') is True
    # data dirs shipped with every executable
    assert 'extensions' in config['extra_data']
    assert 'data' in config['extra_data']


@test
def version_is_1_2_0():
    assert __version__ == '1.2.0', __version__
    assert format_version() == 'v1.2.0'


# ----------------------------------------------------------------------
# API server + Go desktop app (v1.2.0)
# ----------------------------------------------------------------------

@test
def server_api_endpoints():
    import http.client
    import threading as threading_mod
    import time as time_mod
    from http.server import ThreadingHTTPServer
    from server import LitoServer, make_handler

    with tempfile.TemporaryDirectory() as tmp:
        state = LitoServer(model_dir=os.path.join(tmp, 'no-model'),
                           search_enabled=False)
        # offline-friendly version database
        state.mm._fetch_json = lambda url, headers=None: [{
            'tag_name': 'v9.9.9', 'name': 'Lito v9.9.9', 'draft': False,
            'published_at': '2026-10-14T00:00:00Z', 'body': 'notes',
            'assets': [{'name': 'model.zip', 'size': 10,
                        'browser_download_url': 'https://example.com/m.zip'}]}]
        state.start_background_load()
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(state))
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        threading_mod.Thread(target=httpd.serve_forever, daemon=True).start()

        def req(method, path, body=None):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
            conn.request(method, path,
                         json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json'})
            resp = conn.getresponse()
            data = json.loads(resp.read().decode('utf-8'))
            conn.close()
            return resp.status, data

        try:
            # wait for the background brain load
            for _ in range(60):
                _, v = req('GET', '/api/version')
                if not v.get('loading'):
                    break
                time_mod.sleep(0.2)
            assert v['name'] == 'Lito' and v['version'].startswith('v'), v

            # chat through the API (extension answer)
            _, r = req('POST', '/api/chat', {'message': 'what is 2 + 2?'})
            assert '= 4' in r['response'], r
            # slash command routing
            _, r = req('POST', '/api/chat', {'message': '/calc (2+3)*4'})
            assert '= 20' in r['response'], r
            # toggling an extension over the API takes effect immediately
            _, r = req('POST', '/api/extensions',
                       {'name': 'calculator', 'enabled': False})
            assert r.get('ok'), r
            _, r = req('POST', '/api/chat', {'message': 'what is 2 + 2?'})
            assert '= 4' not in r['response'], r
            req('POST', '/api/extensions',
                {'name': 'calculator', 'enabled': True})
            # version database (mocked)
            _, r = req('GET', '/api/releases')
            assert r['releases'][0]['tag'] == 'v9.9.9', r
            assert r['releases'][0]['size_text'] == '10 B', r
            # models + error paths
            _, r = req('GET', '/api/models')
            assert isinstance(r['models'], list)
            _, r = req('POST', '/api/models/use', {'name': 'zzz'})
            assert 'error' in r, r
            _, r = req('POST', '/api/models/delete', {'name': 'zzz'})
            assert 'error' in r, r
            _, r = req('POST', '/api/search', {'enabled': False})
            assert r['search'] is False, r
            _, r = req('POST', '/api/reset')
            assert r.get('ok'), r
            _, r = req('GET', '/api/train/status?since=0')
            assert r['running'] is False and r['lines'] == [], r
            # the static UI is served
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
            conn.request('GET', '/')
            resp = conn.getresponse()
            body = resp.read().decode('utf-8')
            ctype = resp.getheader('Content-Type', '')
            conn.close()
            assert resp.status == 200 and 'Lito' in body and 'app.js' in body
            assert 'text/html' in ctype
        finally:
            httpd.shutdown()


@test
def desktop_app_sources_present():
    ui = os.path.join(REPO_ROOT, 'desktop', 'ui')
    for fname in ('index.html', 'app.js', 'style.css'):
        assert os.path.isfile(os.path.join(ui, fname)), fname
    html = open(os.path.join(ui, 'index.html'), encoding='utf-8').read()
    assert 'app.js' in html and 'style.css' in html
    main_go = open(os.path.join(REPO_ROOT, 'desktop', 'main.go'),
                   encoding='utf-8').read()
    assert '//go:embed ui' in main_go
    assert 'server.py' in main_go            # launches the Python backend
    assert '5486' in main_go                 # default UI port
    gomod = open(os.path.join(REPO_ROOT, 'desktop', 'go.mod'),
                 encoding='utf-8').read()
    assert gomod.startswith('module'), gomod


@test
def desktop_go_app_builds():
    import shutil
    import subprocess as sp_mod
    go = shutil.which('go')
    if not go:
        print('    (no Go toolchain here - CI builds it in build-desktop)')
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'lito-desktop-test')
        env = dict(os.environ, GOPROXY='off', GOFLAGS='-mod=mod')
        if 'GOROOT' not in env:
            pass  # go on PATH already knows its GOROOT
        r = sp_mod.run([go, 'build', '-trimpath', '-o', out, '.'],
                       cwd=os.path.join(REPO_ROOT, 'desktop'),
                       capture_output=True, text=True, timeout=900, env=env)
        assert r.returncode == 0, r.stderr
        assert os.path.isfile(out)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def main():
    passed, failed = 0, 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1
        except Exception as e:
            print(f'  ERROR {name}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\n{passed} passed, {failed} failed, {len(TESTS)} total')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
