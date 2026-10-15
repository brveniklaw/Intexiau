"""Cross-encoder reranking.

Unlike the embeddings, a cross-encoder reads the question and a candidate
section together, so it can judge relevance directly. That makes it too slow to
run over the whole corpus, which is why it only rescores the top candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from bailey_b.config import SearchSettings


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query, 0 to 1, same order as ``texts``."""
        ...


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 via FlagEmbedding. The model loads on first use (~2.3 GB)."""

    def __init__(self, settings: SearchSettings):
        self.name = settings.reranker_model
        self._settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.name, use_fp16=False, devices="cpu")
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().compute_score(
            [[query, t] for t in texts],
            normalize=True,
            max_length=self._settings.rerank_max_tokens,
            batch_size=10,
        )
        if isinstance(scores, float):  # FlagEmbedding unwraps a single pair
            scores = [scores]
        return [float(s) for s in scores]
