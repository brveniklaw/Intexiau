"""Build step 2: embed chunks with bge-m3 (dense + sparse in one pass) and
write them to the vector store (pgvector; Supabase in production).

Indexing is incremental. A chunk is re-embedded only when its text or the
embedding model changed; other field changes (e.g. a new retrieval date after
the annual re-download) are updated in place; chunks that disappeared from the
corpus (repealed or renumbered sections) are deleted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from bailey_b.index.embedder import Embedder
from bailey_b.index.pgvector_store import PgVectorStore
from bailey_b.models import Chunk


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IndexReport:
    embedded: int = 0
    # Text and model unchanged: vectors kept, other fields refreshed.
    unchanged: int = 0
    deleted: int = 0


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    corpus: str,
    store: PgVectorStore,
    embedder: Embedder,
    batch_size: int = 32,
    max_delete_fraction: float = 0.2,
    progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    """Bring ``corpus`` in the store in line with ``chunks``.

    Each batch is committed as it finishes, so an interrupted run resumes where
    it stopped instead of re-embedding everything.

    Refuses to delete more than ``max_delete_fraction`` of the stored corpus, so a
    failed download or parse (an empty or truncated chunk file) cannot silently
    wipe the index. Pass 1.0 to allow it deliberately.
    """
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids")
    stored = store.state(corpus)
    report = IndexReport()

    to_embed: list[tuple[Chunk, str]] = []
    to_update: list[Chunk] = []
    for c in chunks:
        sha = text_sha256(c.text)
        prev = stored.get(c.chunk_id)
        if prev is None or prev.text_sha256 != sha or prev.embedding_model != embedder.name:
            to_embed.append((c, sha))
        else:
            to_update.append(c)

    gone = set(stored) - set(ids)
    if stored and len(gone) > max_delete_fraction * len(stored):
        raise RuntimeError(
            f"refusing to delete {len(gone)} of {len(stored)} stored chunks in {corpus!r}; "
            "check the chunk file, or allow it with a higher max_delete_fraction"
        )
    report.deleted = store.delete(gone)
    report.unchanged = store.update_metadata(corpus, to_update)

    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        embeddings = embedder.encode([c.text for c, _ in batch])
        store.upsert(
            corpus,
            ((c, sha, e) for (c, sha), e in zip(batch, embeddings, strict=True)),
            embedder.name,
        )
        report.embedded += len(batch)
        if progress:
            progress(report.embedded, len(to_embed))
    return report
