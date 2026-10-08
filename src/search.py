"""
Web search - free, key-less search using public APIs, built on the standard
library only.

Sources:
- Wikipedia search + article summaries (en.wikipedia.org/w/api.php and
  /api/rest_v1/page/summary)
- DuckDuckGo Instant Answers (api.duckduckgo.com)

Features:
- No API keys, no extra dependencies (urllib only)
- Disk cache with a time-to-live, so repeated queries are instant
- Short timeouts and an offline cooldown so the chat never stalls when the
  internet is unavailable
- Works fully offline: search() simply returns [] and the caller can fall
  back to built-in knowledge
"""

import hashlib
import html as html_lib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

USER_AGENT = 'Lito/1.0 (github.com/OmniNodeCo/Lito; educational project)'
WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php'
WIKIPEDIA_SUMMARY_API = 'https://en.wikipedia.org/api/rest_v1/page/summary/{}'
DUCKDUCKGO_API = 'https://api.duckduckgo.com/'

DEFAULT_TIMEOUT = 6.0
CACHE_TTL = 7 * 24 * 3600      # one week
OFFLINE_COOLDOWN = 60.0        # seconds to skip network after a failure


@dataclass
class SearchResult:
    """One web search result."""
    title: str
    snippet: str
    url: str
    source: str                  # 'wikipedia' | 'duckduckgo'
    extract: str = ''            # longer summary, when available

    @property
    def display(self) -> str:
        return self.extract if self.extract else self.snippet

    def __str__(self):
        return f"[{self.source}] {self.title} - {self.url}"


def strip_html(text: str) -> str:
    """Remove HTML tags (e.g. Wikipedia highlight spans) and entities."""
    text = re.sub(r'<[^>]+>', '', text or '')
    return html_lib.unescape(text).strip()


class WebSearch:
    """Key-less web search with caching and graceful offline behavior."""

    def __init__(self, cache_dir: str = 'data/cache', timeout: float = DEFAULT_TIMEOUT,
                 enabled: bool = True, cache_ttl: int = CACHE_TTL):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.enabled = enabled
        self.cache_ttl = cache_ttl
        self._cache_path = os.path.join(cache_dir, 'search_cache.json')
        self._cache: Dict[str, dict] = {}
        self._cache_loaded = False
        self._last_failure = 0.0
        # Outcome of the last search: 'ok' | 'offline' | 'no_results' | 'none'
        self.last_status = 'none'
        self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 3, expand: bool = True) -> List[SearchResult]:
        """
        Search the web. Returns up to `limit` results, best first.
        Falls back to the cache, and returns [] when offline (never raises).
        """
        query = (query or '').strip()
        if not query or not self.enabled:
            return []

        key = self._cache_key(query)
        cached = self._cached(key)
        if cached is not None:
            self.last_status = 'ok'
            return cached[:limit]

        if self._network_in_cooldown():
            self.last_status = 'offline'
            return []

        results: List[SearchResult] = []
        # 1. Wikipedia search
        wiki = self._wikipedia_search(query, limit=limit + 2)
        # 2. DuckDuckGo instant answers
        ddg = self._duckduckgo(query)

        if not wiki and not ddg:
            # Distinguish "offline" from "online but nothing found": an empty
            # Wikipedia response with no DDG data usually means a network error.
            self.last_status = 'offline' if self._last_failure else 'no_results'
            if self._last_failure == 0.0:
                # None of the APIs failed explicitly; assume no results.
                self.last_status = 'no_results'
            self._mark_failure()
            return []

        # Merge, deduplicating by normalized title/URL
        seen = set()
        for r in wiki + ddg:
            marker = (r.url or r.title).lower()
            if marker in seen:
                continue
            seen.add(marker)
            results.append(r)

        # Expand the best Wikipedia hit with a full article extract
        if expand and results:
            for r in results:
                if r.source == 'wikipedia' and not r.extract:
                    summary = self._wikipedia_summary(r.title)
                    if summary:
                        r.extract = summary['extract']
                        if summary.get('url'):
                            r.url = summary['url']
                    break

        if not results:
            self.last_status = 'no_results'
            return []

        self.last_status = 'ok'
        self._store(key, results)
        return results[:limit]

    def quick_answer(self, query: str) -> Optional[SearchResult]:
        """Best single result for a query, with the richest text available."""
        results = self.search(query, limit=1)
        return results[0] if results else None

    def is_online(self) -> bool:
        """Quick connectivity check (respects the offline cooldown)."""
        if not self.enabled or self._network_in_cooldown():
            return False
        try:
            params = urllib.parse.urlencode(
                {'action': 'query', 'meta': 'siteinfo', 'format': 'json'})
            req = urllib.request.Request(
                f'{WIKIPEDIA_API}?{params}', headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=min(self.timeout, 3.0)):
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            self._mark_failure()
            return False

    # ------------------------------------------------------------------
    # Wikipedia
    # ------------------------------------------------------------------

    def _wikipedia_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        params = urllib.parse.urlencode({
            'action': 'query', 'list': 'search', 'srsearch': query,
            'srlimit': str(limit), 'format': 'json',
        })
        data = self._get_json(f'{WIKIPEDIA_API}?{params}')
        return self.parse_wikipedia_search(data)

    @staticmethod
    def parse_wikipedia_search(data: Optional[dict]) -> List[SearchResult]:
        if not isinstance(data, dict):
            return []
        results = []
        for item in (data.get('query') or {}).get('search') or []:
            title = (item.get('title') or '').strip()
            snippet = strip_html(item.get('snippet') or '')
            if title:
                results.append(SearchResult(
                    title=title, snippet=snippet,
                    url='https://en.wikipedia.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')),
                    source='wikipedia'))
        return results

    def _wikipedia_summary(self, title: str) -> Optional[dict]:
        url = WIKIPEDIA_SUMMARY_API.format(urllib.parse.quote(title.replace(' ', '_')))
        data = self._get_json(url)
        return self.parse_wikipedia_summary(data)

    @staticmethod
    def parse_wikipedia_summary(data: Optional[dict]) -> Optional[dict]:
        if not isinstance(data, dict) or data.get('type') == 'disambiguation':
            return None
        extract = (data.get('extract') or '').strip()
        if not extract:
            return None
        url = ''
        try:
            url = (data.get('content_urls') or {}).get('desktop', {}).get('page', '')
        except AttributeError:
            pass
        return {'extract': extract, 'url': url, 'title': data.get('title', '')}

    # ------------------------------------------------------------------
    # DuckDuckGo
    # ------------------------------------------------------------------

    def _duckduckgo(self, query: str) -> List[SearchResult]:
        params = urllib.parse.urlencode(
            {'q': query, 'format': 'json', 'no_html': '1', 'skip_disambig': '1'})
        data = self._get_json(f'{DUCKDUCKGO_API}?{params}')
        return self.parse_duckduckgo(data)

    @staticmethod
    def parse_duckduckgo(data: Optional[dict]) -> List[SearchResult]:
        if not isinstance(data, dict):
            return []
        results: List[SearchResult] = []

        abstract = (data.get('AbstractText') or '').strip()
        abstract_url = (data.get('AbstractURL') or '').strip()
        heading = (data.get('Heading') or '').strip()
        if abstract:
            results.append(SearchResult(
                title=heading or 'DuckDuckGo answer', snippet=abstract,
                url=abstract_url, source='duckduckgo', extract=abstract))

        definition = (data.get('Definition') or '').strip()
        if definition:
            results.append(SearchResult(
                title=heading or 'Definition', snippet=definition,
                url=(data.get('DefinitionURL') or '').strip(), source='duckduckgo'))

        for topic in (data.get('Results') or []):
            text = strip_html(topic.get('Text') or '')
            if text and topic.get('FirstURL'):
                results.append(SearchResult(
                    title=text, snippet=text, url=topic['FirstURL'], source='duckduckgo'))

        for topic in (data.get('RelatedTopics') or []):
            text = strip_html(topic.get('Text') or '')
            if text and topic.get('FirstURL'):
                results.append(SearchResult(
                    title=text.split(' - ')[0], snippet=text,
                    url=topic['FirstURL'], source='duckduckgo'))

        return results

    # ------------------------------------------------------------------
    # HTTP + cache plumbing
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> Optional[dict]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode('utf-8', errors='replace'))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, ValueError, OSError):
            self._mark_failure()
            return None

    def _network_in_cooldown(self) -> bool:
        return (time.time() - self._last_failure) < OFFLINE_COOLDOWN

    def _mark_failure(self) -> None:
        self._last_failure = time.time()

    @staticmethod
    def _cache_key(query: str) -> str:
        return hashlib.sha1(query.lower().encode('utf-8')).hexdigest()

    def _load_cache(self) -> None:
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                self._cache_loaded = True
        except (OSError, json.JSONDecodeError, ValueError):
            self._cache = {}

    def _cached(self, key: str) -> Optional[List[SearchResult]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get('ts', 0) > self.cache_ttl:
            return None
        results = []
        for item in entry.get('results', []):
            try:
                results.append(SearchResult(**item))
            except TypeError:
                continue
        return results

    def _store(self, key: str, results: List[SearchResult]) -> None:
        self._cache[key] = {
            'ts': time.time(),
            'results': [{'title': r.title, 'snippet': r.snippet, 'url': r.url,
                         'source': r.source, 'extract': r.extract} for r in results],
        }
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except OSError:
            pass
