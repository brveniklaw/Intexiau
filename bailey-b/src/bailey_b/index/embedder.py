"""Embedding models.

bge-m3 produces a dense vector and a sparse lexical vector (token id -> weight)
in one pass; together they give hybrid search (build doc section 2). Anything
with the same ``encode`` shape can stand in, which is how tests run without
downloading the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bailey_b.config import EmbedSettings

BGE_M3_DENSE_DIM = 1024
# XLM-RoBERTa vocabulary size; sparse vector indices are token ids below this.
BGE_M3_SPARSE_DIM = 250_002
# bge-m3's maximum input length in tokens. Chunks are capped at ~1,200 words,
# well under this, so nothing is truncated.
BGE_M3_MAX_TOKENS = 8192


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]


class Embedder(Protocol):
    name: str
    dense_dim: int
    sparse_dim: int

    def encode(self, texts: Sequence[str]) -> list[Embedding]: ...


class BGEM3Embedder:
    """BAAI/bge-m3 via FlagEmbedding. The model loads on first use (~2.3 GB)."""

    dense_dim = BGE_M3_DENSE_DIM
    sparse_dim = BGE_M3_SPARSE_DIM

    def __init__(self, settings: EmbedSettings):
        self.name = settings.model
        self._settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._settings.model,
                use_fp16=self._settings.use_fp16,
                devices=self._settings.device,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[Embedding]:
        out = self._load().encode(
            list(texts),
            batch_size=self._settings.batch_size,
            max_length=BGE_M3_MAX_TOKENS,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            Embedding(
                dense=[float(x) for x in dense],
                # FlagEmbedding keys lexical weights by token id as a string.
                sparse={int(tok): float(w) for tok, w in weights.items() if w > 0},
            )
            for dense, weights in zip(out["dense_vecs"], out["lexical_weights"], strict=True)
        ]
