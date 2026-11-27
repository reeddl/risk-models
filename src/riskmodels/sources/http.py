"""A small polite HTTP client: per-host rate limits, retries and an on-disk cache.

SEC EDGAR's fair-access policy asks automated clients to stay under 10 requests per second
and to send a User-Agent that identifies them, ideally with a contact email. The default
User-Agent here is generic; set ``RISKMODELS_USER_AGENT`` (for example
``"Your Name your@email.com"``) to declare your own before pulling from EDGAR at scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

DEFAULT_USER_AGENT = "riskmodels/0.1 research client"

# Minimum seconds between requests to the same host.
DEFAULT_MIN_INTERVAL = {
    "www.sec.gov": 0.12,
    "data.sec.gov": 0.12,
    "query1.finance.yahoo.com": 0.5,
    "query2.finance.yahoo.com": 0.5,
    "fred.stlouisfed.org": 0.2,
    "api.finra.org": 0.2,
    "www.aqr.com": 1.0,
}


class Fetcher(Protocol):
    """What the source modules need from a client. Tests pass a fake."""

    def get(self, url: str, params: dict | None = None, max_age_days: float | None = None) -> bytes: ...

    def post_json(self, url: str, body: dict, max_age_days: float | None = None) -> bytes: ...


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int, message: str = "") -> None:
        super().__init__(f"HTTP {status} for {url} {message}".strip())
        self.url = url
        self.status = status


class HttpClient:
    """urllib-based client with caching, per-host throttling and retry on 429/5xx.

    Parameters
    ----------
    cache_dir : path, optional
        Where responses are cached. Defaults to ``$RISKMODELS_CACHE`` or ``~/.cache/riskmodels``.
    user_agent : str, optional
        Defaults to ``$RISKMODELS_USER_AGENT`` or a generic client name.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = None,
        user_agent: str | None = None,
        min_interval: dict[str, float] | None = None,
        max_retries: int = 4,
        timeout: float = 60.0,
    ) -> None:
        root = cache_dir or os.environ.get("RISKMODELS_CACHE") or Path.home() / ".cache" / "riskmodels"
        self.cache_dir = Path(root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or os.environ.get("RISKMODELS_USER_AGENT") or DEFAULT_USER_AGENT
        self.min_interval = {**DEFAULT_MIN_INTERVAL, **(min_interval or {})}
        self.max_retries = max_retries
        self.timeout = timeout
        self._last: dict[str, float] = {}

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / h[:2] / h

    def _read_cache(self, key: str, max_age_days: float | None) -> bytes | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        if max_age_days is not None and time.time() - p.stat().st_mtime > max_age_days * 86400:
            return None
        return p.read_bytes()

    def _write_cache(self, key: str, data: bytes) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    # ------------------------------------------------------------------ network
    def _throttle(self, host: str) -> None:
        gap = self.min_interval.get(host, 0.2)
        wait = self._last.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def _request(self, req: urllib.request.Request) -> bytes:
        host = urllib.parse.urlsplit(req.full_url).hostname or ""
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or 500 <= e.code < 600
                if not retryable or attempt == self.max_retries:
                    raise HttpError(req.full_url, e.code) from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = max(delay, float(retry_after)) if retry_after else delay
                except ValueError:
                    pass
            except urllib.error.URLError:
                if attempt == self.max_retries:
                    raise
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def get(self, url: str, params: dict | None = None, max_age_days: float | None = None) -> bytes:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        cached = self._read_cache(url, max_age_days)
        if cached is not None:
            return cached
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept-Encoding": "identity",
        })
        data = self._request(req)
        self._write_cache(url, data)
        return data

    def post_json(self, url: str, body: dict, max_age_days: float | None = None) -> bytes:
        payload = json.dumps(body, sort_keys=True).encode()
        key = f"POST {url} {payload.decode()}"
        cached = self._read_cache(key, max_age_days)
        if cached is not None:
            return cached
        req = urllib.request.Request(url, data=payload, method="POST", headers={
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        data = self._request(req)
        self._write_cache(key, data)
        return data
