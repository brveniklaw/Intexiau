"""Polite HTTP fetching with an on-disk manifest.

Every raw file lands next to a ``manifest.json`` recording its source URL,
retrieval time and SHA-256, so a refresh can tell what changed and every chunk
can point back to the exact page it came from.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

USER_AGENT = "BaileyB/0.1 (Bay Area Legal Services legal-aid research corpus)"
# Seconds between requests to the same site.
REQUEST_DELAY = 1.0


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(60.0),
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _get(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


def load_manifest(directory: Path) -> dict[str, dict]:
    path = directory / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fetch_all(urls: dict[str, str], directory: Path, *, force: bool = False) -> dict[str, dict]:
    """Download ``{filename: url}`` into ``directory``; skip files already present unless forced.

    Returns the manifest: ``{filename: {url, retrieved_at, sha256, bytes}}``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(directory)
    with _client() as client:
        first = True
        for name, url in urls.items():
            target = directory / name
            if target.exists() and name in manifest and not force:
                continue
            if not first:
                time.sleep(REQUEST_DELAY)
            first = False
            body = _get(client, url)
            target.write_bytes(body)
            manifest[name] = {
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
            # Write after every file so an interrupted run keeps what it fetched.
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
