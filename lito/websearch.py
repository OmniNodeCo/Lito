"""Smarter internet search without heavy deps.

Uses DuckDuckGo Instant Answer API + HTML lite results, then optionally
fetches the top hit for a short extract. Stdlib only.
"""

from __future__ import annotations

import html as html_mod
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


UA = "Lito/0.1.3 (desktop-ai; +https://github.com/OmniNodeCo/Lito)"


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


def _get(url: str, timeout: float = 10.0, max_bytes: int = 200_000) -> tuple[bool, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes)
            ctype = resp.headers.get("Content-Type", "")
            final = resp.geturl()
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}", ""
    except urllib.error.URLError as exc:
        return False, f"Network error: {exc.reason}", ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    return True, text, final


def _strip_tags(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duckduckgo_instant(query: str) -> dict[str, Any] | None:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    ok, body, _ = _get(url, timeout=8, max_bytes=100_000)
    if not ok:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def duckduckgo_html_results(query: str, limit: int = 5) -> list[SearchHit]:
    """Parse DuckDuckGo HTML results page (no API key)."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    ok, body, _ = _get(url, timeout=10)
    if not ok:
        return []
    hits: list[SearchHit] = []
    # result blocks
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        body,
        flags=re.I | re.S,
    ):
        href, title, snip = m.group(1), m.group(2), m.group(3)
        # DDG wraps redirects
        if "uddg=" in href:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = urllib.parse.unquote(q.get("uddg", [href])[0])
        title = _strip_tags(title)
        snip = _strip_tags(snip)
        if not title or not href.startswith("http"):
            continue
        hits.append(SearchHit(title=title, url=href, snippet=snip[:240]))
        if len(hits) >= limit:
            break
    if hits:
        return hits
    # fallback simpler pattern
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, flags=re.I | re.S):
        href, title = m.group(1), _strip_tags(m.group(2))
        if "uddg=" in href:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = urllib.parse.unquote(q.get("uddg", [href])[0])
        if href.startswith("http") and title:
            hits.append(SearchHit(title=title, url=href))
        if len(hits) >= limit:
            break
    return hits


def wikipedia_summary(query: str) -> SearchHit | None:
    api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(query.replace(" ", "_"))
    ok, body, final = _get(api, timeout=6, max_bytes=80_000)
    if not ok:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if data.get("type") == "disambiguation":
        return None
    extract = (data.get("extract") or "").strip()
    title = data.get("title") or query
    url = (data.get("content_urls") or {}).get("desktop", {}).get("page") or final
    if not extract:
        return None
    return SearchHit(title=title, url=url or "", snippet=extract[:500])


def fetch_page_extract(url: str, max_chars: int = 900) -> str:
    ok, body, _ = _get(url, timeout=8, max_bytes=120_000)
    if not ok:
        return ""
    text = _strip_tags(body)
    # Drop very short / cookie walls
    if len(text) < 80:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def smart_search(query: str, *, open_browser: bool = False, limit: int = 5) -> tuple[bool, str]:
    """Return a concise answer + top links. Optionally open the browser too."""
    query = (query or "").strip()
    if not query:
        return False, "What should I search for?"

    lines: list[str] = [f"**Web search:** {query}", ""]

    # 1) Instant answers
    instant = duckduckgo_instant(query)
    answered = False
    if instant:
        abstract = (instant.get("AbstractText") or "").strip()
        heading = (instant.get("Heading") or "").strip()
        abs_url = (instant.get("AbstractURL") or "").strip()
        answer = (instant.get("Answer") or "").strip()
        definition = (instant.get("Definition") or "").strip()
        if answer:
            lines.append(f"**Answer:** {answer}")
            answered = True
        if abstract:
            title = heading or "Summary"
            lines.append(f"**{title}**" + (f" - {abs_url}" if abs_url else ""))
            lines.append(abstract[:600] + ("..." if len(abstract) > 600 else ""))
            answered = True
        elif definition:
            lines.append(f"**Definition:** {definition[:500]}")
            answered = True
        # Related topics
        related = instant.get("RelatedTopics") or []
        rel_lines = []
        for item in related[:4]:
            if isinstance(item, dict) and item.get("Text"):
                rel_lines.append(f"- {item['Text'][:160]}" + (f" ({item.get('FirstURL')})" if item.get("FirstURL") else ""))
            elif isinstance(item, dict) and item.get("Topics"):
                for sub in item["Topics"][:2]:
                    if isinstance(sub, dict) and sub.get("Text"):
                        rel_lines.append(f"- {sub['Text'][:160]}")
        if rel_lines:
            lines.append("\n**Related:**")
            lines.extend(rel_lines)
            answered = True

    # 2) Wikipedia short summary if still thin
    if not answered:
        wiki = wikipedia_summary(query)
        if wiki:
            lines.append(f"**{wiki.title}** (Wikipedia)")
            lines.append(wiki.snippet)
            if wiki.url:
                lines.append(wiki.url)
            answered = True

    # 3) Top web results
    hits = duckduckgo_html_results(query, limit=limit)
    if hits:
        lines.append("\n**Top results:**")
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. **{h.title}**")
            if h.snippet:
                lines.append(f"   {h.snippet}")
            lines.append(f"   {h.url}")
        # 4) Fetch first hit extract for smarter brief
        if not answered and hits[0].url:
            extract = fetch_page_extract(hits[0].url)
            if extract:
                lines.append(f"\n**From top result ({hits[0].title}):**")
                lines.append(extract)
                answered = True
    elif not answered:
        lines.append("No results (offline or blocked). Try again, or `open https://duckduckgo.com`.")
        return False, "\n".join(lines)

    ddg = "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query})
    lines.append(f"\nFull page: {ddg}")
    lines.append("Say `open <url>` to open a result, or `fetch <url>` to read more.")

    if open_browser:
        try:
            from .apps import open_url

            open_url(ddg)
        except Exception:
            pass

    return True, "\n".join(lines)
