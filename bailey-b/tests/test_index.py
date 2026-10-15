"""Incremental indexing against a real Postgres + pgvector, with a stand-in embedder.

Set BAILEY_TEST_POSTGRES_DSN to run (skipped otherwise). The test creates and
drops its own table, so it is safe to point at a dev database.
"""

import hashlib
import os
import uuid

import pytest

from bailey_b.index import index_chunks
from bailey_b.index.embedder import Embedding
from bailey_b.models import Chunk

DSN = os.environ.get("BAILEY_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="BAILEY_TEST_POSTGRES_DSN not set")


class HashEmbedder:
    """Deterministic vectors derived from the text; records what it was asked to encode."""

    dense_dim = 8
    sparse_dim = 100

    def __init__(self, name: str = "hash-v1"):
        self.name = name
        self.seen: list[str] = []

    def encode(self, texts):
        self.seen.extend(texts)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append(Embedding(dense=[b / 255 + 0.01 for b in h[:8]], sparse={h[0] % 100: 1.0}))
        return out


def chunk(section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"fl-stat:{section}#0",
        citation_id=f"fl-stat:{section}",
        display_citation=f"Fla. Stat. § {section}",
        text=text,
        metadata={"chapter": section.split(".")[0]},
    )


@pytest.fixture
def store():
    from bailey_b.index.pgvector_store import PgVectorStore

    s = PgVectorStore(
        DSN,
        f"test_{uuid.uuid4().hex[:8]}",
        dense_dim=HashEmbedder.dense_dim,
        sparse_dim=HashEmbedder.sparse_dim,
    )
    s.ensure_schema()
    yield s
    s.drop()
    s.close()


def test_incremental_indexing(store):
    emb = HashEmbedder()
    v1 = [chunk("83.56", "termination"), chunk("83.57", "tenancy"), chunk("83.58", "holdover")]

    r = index_chunks(v1, corpus="fl-statutes", store=store, embedder=emb)
    assert (r.embedded, r.unchanged, r.deleted) == (3, 0, 0)
    assert store.counts() == {"fl-statutes": (3, 3)}

    # Same corpus again: nothing re-embedded.
    emb.seen.clear()
    r = index_chunks(v1, corpus="fl-statutes", store=store, embedder=emb)
    assert (r.embedded, r.unchanged, r.deleted) == (0, 3, 0)
    assert emb.seen == []

    # One section amended, one repealed: only the amended one is re-embedded.
    v2 = [chunk("83.56", "termination, amended"), chunk("83.57", "tenancy")]
    r = index_chunks(v2, corpus="fl-statutes", store=store, embedder=emb, max_delete_fraction=0.5)
    assert (r.embedded, r.unchanged, r.deleted) == (1, 1, 1)
    assert emb.seen == ["termination, amended"]
    assert store.counts() == {"fl-statutes": (2, 2)}

    # New embedding model: everything re-embedded.
    r = index_chunks(v2, corpus="fl-statutes", store=store, embedder=HashEmbedder("hash-v2"))
    assert r.embedded == 2


def test_refuses_mass_deletion(store):
    emb = HashEmbedder()
    chunks = [chunk(f"83.{n}", f"text {n}") for n in range(10, 20)]
    index_chunks(chunks, corpus="fl-statutes", store=store, embedder=emb)
    with pytest.raises(RuntimeError, match="refusing to delete"):
        index_chunks(chunks[:5], corpus="fl-statutes", store=store, embedder=emb)
    assert store.counts() == {"fl-statutes": (10, 10)}
    r = index_chunks(
        chunks[:5], corpus="fl-statutes", store=store, embedder=emb, max_delete_fraction=1.0
    )
    assert r.deleted == 5


def test_corpora_are_independent(store):
    emb = HashEmbedder()
    index_chunks([chunk("83.56", "a")], corpus="fl-statutes", store=store, embedder=emb)
    index_chunks([], corpus="other", store=store, embedder=emb)
    assert store.counts() == {"fl-statutes": (1, 1)}


def test_vectors_round_trip(store):
    emb = HashEmbedder()
    index_chunks([chunk("83.56", "termination")], corpus="fl-statutes", store=store, embedder=emb)
    (expected,) = emb.encode(["termination"])
    dense, sparse = store.conn.execute(f"SELECT dense, sparse FROM {store.table}").fetchone()
    assert dense.to_list() == pytest.approx(expected.dense, rel=1e-6)
    assert dict(zip(sparse.indices(), sparse.values(), strict=True)) == expected.sparse


def test_search_dense_sparse_and_citation_ids(store):
    emb = HashEmbedder()
    chunks = [chunk("83.56", "termination"), chunk("83.60", "defenses"), chunk("83.62", "writ")]
    index_chunks(chunks, corpus="fl-statutes", store=store, embedder=emb)
    (q,) = emb.encode(["defenses"])

    dense = store.search_dense(q.dense, 3)
    assert dense[0][0].chunk_id == "fl-stat:83.60#0"
    assert dense[0][1] == pytest.approx(1.0, abs=1e-5)  # identical vector
    assert dense[0][0].metadata == {"chapter": "83"}

    sparse = store.search_sparse(q.sparse, 3)
    assert sparse[0][0].chunk_id == "fl-stat:83.60#0"
    assert all(score > 0 for _, score in sparse)  # non-matching chunks are left out
    assert store.search_sparse({}, 3) == []

    assert store.citation_ids() == {"fl-stat:83.56", "fl-stat:83.60", "fl-stat:83.62"}
