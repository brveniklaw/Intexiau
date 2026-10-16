"""Build step 3: hybrid search + reranking.

``retrieve`` is the single retrieval entry point (build doc section 5). Every
consumer (intake triage, research, draft analysis, deadline tracking, the eval
harness) calls it, and nothing in it depends on which LLM sits downstream.

Query path:

1. Embed the question with bge-m3 (dense + sparse).
2. Fetch ``pool`` chunks by meaning and ``pool`` by keywords.
3. Fuse the two lists with reciprocal rank fusion (RRF): each list contributes
   ``1 / (rrf_k + rank)``, so the very different score scales never need
   reconciling, and a chunk both lists like rises to the top.
4. Collapse to one chunk per citation (a long section split into pieces counts
   once, represented by its best-ranked piece), keep the top ``top_k``.
5. Rerank those with a cross-encoder; return the best ``top_n``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from bailey_b.config import SearchSettings, Settings, get_settings
from bailey_b.index.embedder import Embedder
from bailey_b.models import Chunk, RetrievedChunk
from bailey_b.search.reranker import Reranker


class Mode(StrEnum):
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass
class Candidate:
    chunk: Chunk
    fused: float = 0.0
    dense: float | None = None
    sparse: float | None = None
    rerank: float | None = None
    # Blended rerank + fusion-rank score that sets the final order (see blend()).
    final: float | None = None
    # Distinct pieces of the same section that matched (a split section counts once).
    pieces: list[str] = field(default_factory=list)


def fuse(
    dense: Sequence[tuple[Chunk, float]],
    sparse: Sequence[tuple[Chunk, float]],
    *,
    rrf_k: int = 60,
    sparse_weight: float = 1.0,
) -> list[Candidate]:
    """Reciprocal rank fusion of two ranked lists, best first."""
    by_id: dict[str, Candidate] = {}
    for results, attr, weight in ((dense, "dense", 1.0), (sparse, "sparse", sparse_weight)):
        for rank, (chunk, score) in enumerate(results, start=1):
            cand = by_id.setdefault(chunk.chunk_id, Candidate(chunk))
            cand.fused += weight / (rrf_k + rank)
            setattr(cand, attr, score)
    return sorted(by_id.values(), key=lambda c: c.fused, reverse=True)


def collapse(candidates: Sequence[Candidate]) -> list[Candidate]:
    """One candidate per citation, keeping the first (best) piece and noting the others."""
    out: dict[str, Candidate] = {}
    for cand in candidates:
        cite = cand.chunk.citation_id
        if cite in out:
            out[cite].pieces.append(cand.chunk.chunk_id)
        else:
            cand.pieces = [cand.chunk.chunk_id]
            out[cite] = cand
    return list(out.values())


def blend(candidates: Sequence[Candidate], rerank_weight: float) -> list[Candidate]:
    """Order reranked candidates by a mix of reranker score and fusion rank.

    ``candidates`` must be in fusion order with ``rerank`` set. The fusion-rank
    score runs linearly from 1.0 (first) toward 0, so a weight below 1.0 keeps
    the reranker from burying a candidate both searches ranked highly.
    """
    n = len(candidates)
    for i, cand in enumerate(candidates):
        rank_score = 1.0 - i / n
        cand.final = rerank_weight * (cand.rerank or 0.0) + (1 - rerank_weight) * rank_score
    return sorted(candidates, key=lambda c: c.final, reverse=True)


class Retriever:
    def __init__(
        self,
        store,
        embedder: Embedder,
        reranker: Reranker | None,
        settings: SearchSettings,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        # Question -> embedding, so comparing modes on one question embeds it once.
        self._embedded: dict[str, object] = {}

    def candidates(self, question: str, mode: Mode = Mode.HYBRID) -> list[Candidate]:
        """Steps 1-4: fused, collapsed candidates before reranking, best first."""
        s = self.settings
        emb = self._embedded.get(question)
        if emb is None:
            (emb,) = self.embedder.encode([question])
            if len(self._embedded) > 256:
                self._embedded.clear()
            self._embedded[question] = emb
        dense = self.store.search_dense(emb.dense, s.pool) if mode != Mode.SPARSE else []
        sparse = self.store.search_sparse(emb.sparse, s.pool) if mode != Mode.DENSE else []
        fused = fuse(dense, sparse, rrf_k=s.rrf_k, sparse_weight=s.sparse_weight)
        return collapse(fused)[: s.top_k]

    def retrieve(
        self,
        question: str,
        *,
        top_n: int | None = None,
        mode: Mode = Mode.HYBRID,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        top_n = top_n or self.settings.top_n
        rerank = self.settings.rerank if rerank is None else rerank
        cands = self.candidates(question, mode)
        if rerank and self.reranker is not None and cands:
            scores = self.reranker.score(question, [c.chunk.text for c in cands])
            for cand, score in zip(cands, scores, strict=True):
                cand.rerank = score
            cands = blend(cands, self.settings.rerank_weight)
        return [
            RetrievedChunk(
                chunk=c.chunk,
                rank=i,
                score=c.final if c.final is not None else c.fused,
                dense_score=c.dense,
                sparse_score=c.sparse,
            )
            for i, c in enumerate(cands[:top_n], start=1)
        ]


_default: Retriever | None = None


def default_retriever(settings: Settings | None = None) -> Retriever:
    """The production wiring: bge-m3, pgvector, bge-reranker. Built once, then reused."""
    global _default
    if _default is None or settings is not None:
        from bailey_b.index.embedder import BGEM3Embedder
        from bailey_b.index.pgvector_store import PgVectorStore
        from bailey_b.search.reranker import BGEReranker

        settings = settings or get_settings()
        embedder = BGEM3Embedder(settings.embed)
        store = PgVectorStore(
            settings.vector_store.postgres_dsn.get_secret_value(),
            settings.vector_store.collection,
            dense_dim=embedder.dense_dim,
            sparse_dim=embedder.sparse_dim,
        )
        _default = Retriever(store, embedder, BGEReranker(settings.search), settings.search)
    return _default


def retrieve(
    question: str,
    *,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return the best ``top_n`` chunks for ``question``, ranked, with citations.

    With reranking on, scores run 0 to 1 (reranker relevance, blended with the
    fusion rank when ``rerank_weight`` < 1); otherwise they are fused rank scores.
    Each result is a distinct citation.
    """
    return default_retriever(settings).retrieve(question, top_n=top_n)
