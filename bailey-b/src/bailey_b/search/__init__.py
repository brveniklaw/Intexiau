"""Build step 3: hybrid search + reranking.

``retrieve`` is the single retrieval entry point (build doc section 5). Every
consumer (intake triage, research, draft analysis, deadline tracking, the eval
harness) calls it, and nothing in it depends on which LLM sits downstream.
"""

from __future__ import annotations

from bailey_b.config import Settings, get_settings
from bailey_b.models import RetrievedChunk


def retrieve(
    question: str,
    *,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return the best ``top_n`` chunks for ``question``, ranked, with citations.

    Query path: embed the question, hybrid (dense + sparse) search for
    ``settings.search.top_k`` candidates, rerank with a cross-encoder, keep ``top_n``.
    """
    settings = settings or get_settings()
    raise NotImplementedError("retrieval lands in build step 3")
