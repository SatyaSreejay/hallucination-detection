"""Evidence retrieval from the official MediaWiki Action API (English Wikipedia).

* identifies itself with a User-Agent (Wikimedia policy),
* applies timeouts, retry with exponential backoff, and a minimum interval between calls,
* caches successful API responses in memory and on disk,
* returns an explicit :class:`RetrievalStatus`; an empty search is ``NO_RESULTS``, which
  callers must treat as "unknown", never as evidence that a claim is false.

Retrieved text is untrusted data and is only ever passed on as evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.retrieval.passages import chunk_text, rank_passages, subject_related, tokenize
from src.schemas import EvidencePassage, RetrievalResult, RetrievalStatus

logger = logging.getLogger(__name__)

MAX_EXTRACT_CHARS = 60_000


class WikipediaAPIError(RuntimeError):
    """Network/HTTP/format failure talking to the MediaWiki API."""


class WikipediaRetriever:
    """Search Wikipedia for a claim and return the most relevant passages."""

    source_name = "wikipedia"

    def __init__(
        self,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str = "HallucinationDetectionPrototype/0.1",
        timeout_s: float = 10.0,
        max_retries: int = 3,
        search_results: int = 3,
        passages_per_claim: int = 3,
        min_interval_s: float = 0.2,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        self.api_url = api_url
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self.search_results = search_results
        self.passages_per_claim = passages_per_claim
        self.min_interval_s = min_interval_s
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._sleep = sleep
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._last_request = 0.0
        self.request_count = 0

    # ----------------------------------------------------------------- HTTP
    def _cache_key(self, params: dict[str, Any]) -> str:
        blob = json.dumps(sorted(params.items()), ensure_ascii=False)
        return hashlib.sha256(f"{self.api_url}|{blob}".encode()).hexdigest()

    def _get(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """GET the API with caching and retries. Returns ``(json, from_cache)``."""
        params = {**params, "format": "json", "formatversion": 2}
        key = self._cache_key(params)
        if key in self._memory_cache:
            return self._memory_cache[key], True
        disk_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if disk_path and disk_path.is_file():
            try:
                data = json.loads(disk_path.read_text(encoding="utf-8"))
                self._memory_cache[key] = data
                return data, True
            except (OSError, ValueError):
                logger.warning("Ignoring unreadable cache file %s", disk_path)

        last_error: str = "unknown error"
        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            self.request_count += 1
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:300]
            except requests.RequestException as exc:
                raise WikipediaAPIError(f"Request failed: {exc}") from exc
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise WikipediaAPIError("Wikipedia returned non-JSON data") from exc
                    if "error" in data:
                        raise WikipediaAPIError(f"MediaWiki API error: {data['error']}")
                    self._memory_cache[key] = data
                    if disk_path:
                        try:
                            disk_path.parent.mkdir(parents=True, exist_ok=True)
                            disk_path.write_text(json.dumps(data), encoding="utf-8")
                        except OSError:
                            logger.warning("Could not write cache file %s", disk_path)
                    return data, False
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise WikipediaAPIError(last_error)
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(min(float(retry_after), 10.0))
            if attempt < self.max_retries - 1:
                self._sleep(0.5 * 2**attempt)
        raise WikipediaAPIError(f"Wikipedia unreachable after {self.max_retries} attempts ({last_error})")

    # ------------------------------------------------------------ retrieval
    def search(self, query: str) -> tuple[list[dict[str, Any]], bool]:
        data, cached = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query[:300],
                "srlimit": self.search_results,
                "srprop": "",
            }
        )
        return data.get("query", {}).get("search", []), cached

    def fetch_page(self, page_id: int) -> tuple[dict[str, Any] | None, bool]:
        data, cached = self._get(
            {
                "action": "query",
                "prop": "extracts|info",
                "pageids": page_id,
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        page = pages[0] if pages else None
        if not page or page.get("missing") or not page.get("extract"):
            return None, cached
        return page, cached

    accepts_topic = True  # the pipeline may pass the answer's subject (see HallucinationPipeline)

    def top_title(self, text: str) -> str | None:
        """Title of the best search hit for ``text`` (used to find an answer's subject)."""
        try:
            hits, _ = self.search(text.strip()[:300])
        except WikipediaAPIError:
            return None
        return hits[0]["title"] if hits else None

    def retrieve(self, claim_text: str, topic: str | None = None) -> RetrievalResult:
        """Return the top passages for ``claim_text`` with an explicit status.

        ``topic`` is the subject of the whole answer. It is added to the search query when
        the claim does not name it ("It was designated a UNESCO site in 1983"), so the
        search finds the right article instead of any page that matches the words.
        """
        query = claim_text.strip()
        if topic and not subject_related(topic, query):
            query = f"{topic}: {query}"
        try:
            hits, cached = self.search(query)
            if not hits:
                # fall back to a keyword query (long sentences can return nothing)
                keywords = " ".join(dict.fromkeys(tokenize(query)))[:200]
                if keywords and keywords != query.lower():
                    hits, cached = self.search(keywords)
            if topic:
                # Search needs every word to match, so detailed claims ("18,038 iron pieces
                # weighing 10,100 tonnes") often find nothing. The answer's own article is
                # therefore always a candidate; passage ranking picks the relevant part.
                topic_hits, topic_cached = self.search(topic)
                cached = cached and topic_cached
                if topic_hits:
                    hits = topic_hits[:1] + [h for h in hits if h["pageid"] != topic_hits[0]["pageid"]]
            if not hits:
                return RetrievalResult(query, RetrievalStatus.NO_RESULTS, from_cache=cached)

            candidates: list[tuple[dict[str, Any], int, str]] = []
            all_cached = cached
            for hit in hits[: self.search_results + bool(topic)]:
                page, page_cached = self.fetch_page(hit["pageid"])
                all_cached = all_cached and page_cached
                if page is None:
                    continue
                for i, passage in enumerate(chunk_text(page["extract"][:MAX_EXTRACT_CHARS])):
                    candidates.append((page, i, passage))
        except WikipediaAPIError as exc:
            return RetrievalResult(query, RetrievalStatus.ERROR, error=str(exc))
        except (KeyError, TypeError) as exc:
            return RetrievalResult(
                query, RetrievalStatus.ERROR, error=f"Unexpected API response format: {exc}"
            )

        # rank with the page title prepended so passages using pronouns still match
        ranked = rank_passages(
            query, [f"{p['title']} {text}" for p, _, text in candidates], self.passages_per_claim
        )
        passages = [
            EvidencePassage(
                title=candidates[i][0]["title"],
                url=candidates[i][0].get("fullurl")
                or "https://en.wikipedia.org/?curid=" + str(candidates[i][0]["pageid"]),
                text=candidates[i][2],
                source=self.source_name,
                page_id=candidates[i][0].get("pageid"),
                passage_index=candidates[i][1],
                retrieval_score=round(score, 4),
            )
            for i, score in ranked
        ]
        status = RetrievalStatus.OK if passages else RetrievalStatus.NO_RESULTS
        return RetrievalResult(query, status, passages, from_cache=all_cached)

    def check_connectivity(self) -> tuple[bool, str]:
        """Cheap live check used by the dashboard status panel."""
        try:
            self._get({"action": "query", "meta": "siteinfo", "siprop": "general"})
            return True, "Wikipedia API reachable"
        except WikipediaAPIError as exc:
            return False, str(exc)
