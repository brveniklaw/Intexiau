"""Fusion, collapsing and the retrieval flow, with stand-ins for the models and store."""

import pytest

from bailey_b.config import SearchSettings
from bailey_b.index.embedder import Embedding
from bailey_b.models import Chunk
from bailey_b.search import Mode, Retriever, collapse, fuse


def chunk(section: str, piece: int = 0, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"fl-stat:{section}#{piece}",
        citation_id=f"fl-stat:{section}",
        display_citation=f"Fla. Stat. § {section}",
        text=text or f"text of {section} piece {piece}",
    )


def test_fuse_rewards_agreement():
    a, b, c = chunk("83.56"), chunk("83.60"), chunk("83.49")
    dense = [(a, 0.9), (b, 0.8)]
    sparse = [(b, 0.3), (c, 0.2)]
    fused = fuse(dense, sparse, rrf_k=60)
    # b is in both lists, so it beats a (dense #1 only).
    assert [f.chunk.chunk_id for f in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].dense == 0.8 and fused[0].sparse == 0.3
    assert fused[1].sparse is None
    assert fused[0].fused == pytest.approx(1 / 62 + 1 / 61)


def test_collapse_keeps_best_piece_per_section():
    cands = fuse([(chunk("61.13", 2), 0.9), (chunk("61.30"), 0.8), (chunk("61.13", 0), 0.7)], [])
    out = collapse(cands)
    assert [c.chunk.chunk_id for c in out] == ["fl-stat:61.13#2", "fl-stat:61.30#0"]
    assert out[0].pieces == ["fl-stat:61.13#2", "fl-stat:61.13#0"]


class FakeEmbedder:
    name = "fake"
    dense_dim = 2
    sparse_dim = 10

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        return [Embedding(dense=[1.0, 0.0], sparse={1: 1.0}) for _ in texts]


class FakeStore:
    def __init__(self, dense, sparse):
        self.dense, self.sparse = dense, sparse

    def search_dense(self, vector, limit):
        return self.dense[:limit]

    def search_sparse(self, weights, limit):
        return self.sparse[:limit]


class FakeReranker:
    """Scores by a fixed per-section table, to check reordering."""

    name = "fake-reranker"

    def __init__(self, table):
        self.table = table
        self.seen = []

    def score(self, query, texts):
        self.seen.append(list(texts))
        return [self.table.get(t.split()[2], 0.0) for t in texts]


@pytest.fixture
def parts():
    a, b, c, d = chunk("83.56"), chunk("83.60"), chunk("83.49"), chunk("83.62")
    store = FakeStore(dense=[(a, 0.9), (c, 0.8), (d, 0.7)], sparse=[(b, 0.4), (a, 0.3)])
    return store, FakeEmbedder()


def test_retrieve_reranks_and_cuts(parts):
    store, emb = parts
    reranker = FakeReranker({"83.60": 0.95, "83.56": 0.7, "83.49": 0.1, "83.62": 0.2})
    r = Retriever(store, emb, reranker, SearchSettings(top_k=20, top_n=2))
    out = r.retrieve("question")
    assert [x.chunk.citation_id for x in out] == ["fl-stat:83.60", "fl-stat:83.56"]
    assert [x.rank for x in out] == [1, 2]
    assert out[0].score == 0.95
    assert out[1].dense_score == 0.9 and out[1].sparse_score == 0.3


def test_retrieve_without_rerank_uses_fusion_order(parts):
    store, emb = parts
    r = Retriever(store, emb, FakeReranker({}), SearchSettings(top_n=3, rerank=False))
    out = r.retrieve("question")
    assert out[0].chunk.citation_id == "fl-stat:83.56"  # in both lists


def test_modes_and_embedding_cache(parts):
    store, emb = parts
    r = Retriever(store, emb, None, SearchSettings(top_n=5))
    dense = r.retrieve("q", mode=Mode.DENSE)
    sparse = r.retrieve("q", mode=Mode.SPARSE)
    assert {x.chunk.citation_id for x in dense} == {
        "fl-stat:83.56",
        "fl-stat:83.49",
        "fl-stat:83.62",
    }
    assert {x.chunk.citation_id for x in sparse} == {"fl-stat:83.60", "fl-stat:83.56"}
    assert emb.calls == 1  # the question was embedded once


def test_reranker_sees_at_most_top_k(parts):
    store, emb = parts
    reranker = FakeReranker({})
    Retriever(store, emb, reranker, SearchSettings(top_k=2)).retrieve("q")
    assert len(reranker.seen[0]) == 2
