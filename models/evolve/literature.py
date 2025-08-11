import datetime as _dt
from typing import List, Dict, Optional

# Prefer lightweight arxiv client; gracefully fallback to HTTP if missing
try:
    import arxiv  # type: ignore
    _HAS_ARXIV = True
except Exception:
    _HAS_ARXIV = False

import logging
import re
import requests
import json
import os
import time
import random

# Global knobs (no hardcoded literals in logic)
ARXIV_MAX_RETRIES: int = int(os.environ.get("ARXIV_MAX_RETRIES", "2"))
ARXIV_BACKOFF_BASE_SEC: float = float(os.environ.get("ARXIV_BACKOFF_BASE_SEC", "1.0"))
ARXIV_BACKOFF_JITTER_SEC: float = float(os.environ.get("ARXIV_BACKOFF_JITTER_SEC", "0.25"))


class ArxivMiner:
    """Fetch recent AI papers from arXiv with simple filters.

    Primary fields extracted: id, title, authors, categories, published, summary (abstract), pdf_url.
    """

    def __init__(self, max_results: int = 50, categories: Optional[List[str]] = None, logger: Optional[logging.Logger] = None):
        self.max_results = max_results
        self.categories = categories or [
            # Core AI/ML categories
            "cs.AI", "cs.LG", "cs.CL", "cs.IR", "stat.ML"
        ]
        self.log = logger or logging.getLogger(__name__)

    def _search_query(self, query_terms: List[str]) -> str:
        # arXiv search syntax: (ti:term OR abs:term) AND (cat:cs.LG OR cat:cs.AI)
        terms = []
        for t in query_terms:
            t = re.sub(r"[\s]+", "+", t.strip())
            terms.append(f"(ti:{t} OR abs:{t})")
        q_terms = " OR ".join(terms) if terms else "(ti:reasoning OR abs:reasoning)"
        q_cats = " OR ".join([f"cat:{c}" for c in self.categories])
        return f"({q_terms}) AND ({q_cats})"

    def fetch(self, query_terms: Optional[List[str]] = None, days_back: int = 14) -> List[Dict]:
        query = self._search_query(query_terms or [
            "reasoning", "planning", "hierarchical", "maze", "sudoku", "chain-of-thought",
            "recurrent transformer", "world model", "ponder", "adaptive computation time",
        ])
        since = _dt.datetime.utcnow() - _dt.timedelta(days=days_back)
        self.log.info("Fetching arXiv papers since %s with query=%s", since.date(), query)

        # Retry wrapper
        last_err: Optional[Exception] = None
        for attempt in range(ARXIV_MAX_RETRIES + 1):
            try:
                if _HAS_ARXIV:
                    papers = self._fetch_with_arxiv_lib(query, since)
                else:
                    papers = self._fetch_with_http(query, since)
                # Filter already-seen papers if env file is provided (read env at call time)
                seen_path = os.environ.get("HRM_SEEN_PAPERS_PATH", "")
                seen_ids = _load_seen_ids(seen_path) if seen_path else set()
                if seen_ids:
                    papers = [p for p in papers if p.get("id") not in seen_ids]
                return papers
            except Exception as e:
                last_err = e
                if attempt >= ARXIV_MAX_RETRIES:
                    break
                backoff = (ARXIV_BACKOFF_BASE_SEC * (2 ** attempt)) + random.random() * ARXIV_BACKOFF_JITTER_SEC
                time.sleep(backoff)
        # Exhausted retries
        if last_err:
            raise last_err
        return []

    def _fetch_with_arxiv_lib(self, query: str, since: _dt.datetime) -> List[Dict]:
        results = []
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for r in search.results():  # type: ignore[attr-defined]
            if r.published.replace(tzinfo=None) < since:
                continue
            results.append({
                "id": r.get_short_id(),
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "categories": list(getattr(r, "categories", []) or []),
                "published": r.published.isoformat(),
                "summary": r.summary,
                "pdf_url": getattr(r, "pdf_url", None),
            })
        return results

    def _fetch_with_http(self, query: str, since: _dt.datetime) -> List[Dict]:
        # Use arXiv API (Atom). We keep it simple to avoid extra deps.
        # Note: arXiv atom date parsing is omitted; we rely on recency via max_results.
        base = "https://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        text = r.text
        # Very light extraction; for robust parsing one should use feedparser.
        entries = text.split("<entry>")[1:]
        out: List[Dict] = []
        for e in entries:
            def _tag(tag: str) -> Optional[str]:
                m = re.search(fr"<{tag}>(.*?)</{tag}>", e, re.DOTALL)
                return m.group(1).strip() if m else None
            title = _tag("title") or ""
            summary = _tag("summary") or ""
            id_tag = _tag("id") or ""
            pdf_url = None
            m = re.search(r"<link[^>]+title=\"pdf\"[^>]+href=\"(.*?)\"", e)
            if m:
                pdf_url = m.group(1)
            out.append({
                "id": id_tag.rsplit("/", 1)[-1],
                "title": re.sub(r"\s+", " ", title),
                "authors": [],
                "categories": [],
                "published": _tag("published") or "",
                "summary": re.sub(r"\s+", " ", summary),
                "pdf_url": pdf_url,
            })
        return out


def _load_seen_ids(path: str) -> set:
    ids = set()
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and "id" in obj:
                            ids.add(obj["id"])
                    except Exception:
                        continue
    except Exception:
        return set()
    return ids
