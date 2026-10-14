"""Retrieval eval harness (build doc section 7).

The eval set is a JSONL file of real BALS questions, each with a known-good
answer and the canonical citation IDs a correct retrieval must surface. It is
built in parallel with corpus ingestion and measured against retrieval output
from day one: if retrieval misses the right chunks, no model choice saves you.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from bailey_b.models import RetrievedChunk, parse_citation_id

MIN_EVAL_QUESTIONS = 20


class PracticeArea(StrEnum):
    FAMILY = "family"
    EVICTION = "eviction"
    CONSUMER = "consumer"
    FORECLOSURE = "foreclosure"
    PROBATE = "probate"
    IMMIGRATION = "immigration"
    BENEFITS = "benefits"
    OTHER = "other"


class EvalQuestion(BaseModel):
    id: str
    question: str
    practice_area: PracticeArea
    expected_citations: list[str] = Field(
        min_length=1,
        description="Canonical citation IDs retrieval must surface, e.g. 'fl-stat:83.56'.",
    )
    answer: str = Field(description="Known-good answer, reviewed by a BALS attorney.")
    reviewed_by: str = Field(description="Initials of the reviewing attorney.")
    notes: str = ""

    @field_validator("expected_citations")
    @classmethod
    def _valid_citations(cls, v: list[str]) -> list[str]:
        for cid in v:
            parse_citation_id(cid)
        return v


def load_eval_set(path: Path) -> list[EvalQuestion]:
    """Load and validate a JSONL eval file. Raises ValueError on bad lines or duplicate IDs."""
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = EvalQuestion.model_validate(json.loads(line))
        except Exception as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
        if q.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {q.id!r}")
        seen.add(q.id)
        questions.append(q)
    return questions


def recall_at_k(expected: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of expected citation IDs that appear in the top k retrieved."""
    expected = set(expected)
    if not expected:
        return 1.0
    return len(expected & set(retrieved[:k])) / len(expected)


def reciprocal_rank(expected: Iterable[str], retrieved: Sequence[str]) -> float:
    """1 / rank of the first expected citation retrieved, or 0 if none were."""
    expected = set(expected)
    for i, cid in enumerate(retrieved, start=1):
        if cid in expected:
            return 1.0 / i
    return 0.0


class QuestionResult(BaseModel):
    id: str
    practice_area: PracticeArea
    retrieved: list[str]
    missed: list[str]
    recall_at_k: float
    reciprocal_rank: float


class EvalReport(BaseModel):
    k: int
    n: int
    mean_recall_at_k: float
    mrr: float
    by_area: dict[str, float] = Field(description="Mean recall@k per practice area.")
    results: list[QuestionResult]


RetrieveFn = Callable[[str], list[RetrievedChunk]]


def run_eval(questions: Sequence[EvalQuestion], retrieve: RetrieveFn, k: int) -> EvalReport:
    """Run every question through ``retrieve`` and score it at cutoff ``k``."""
    results: list[QuestionResult] = []
    for q in questions:
        # Several chunks can share one citation (long opinions); score unique citations in order.
        retrieved = list(dict.fromkeys(rc.chunk.citation_id for rc in retrieve(q.question)))
        results.append(
            QuestionResult(
                id=q.id,
                practice_area=q.practice_area,
                retrieved=retrieved[:k],
                missed=[c for c in q.expected_citations if c not in retrieved[:k]],
                recall_at_k=recall_at_k(q.expected_citations, retrieved, k),
                reciprocal_rank=reciprocal_rank(q.expected_citations, retrieved),
            )
        )

    by_area: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_area[r.practice_area].append(r.recall_at_k)
    n = len(results)
    return EvalReport(
        k=k,
        n=n,
        mean_recall_at_k=sum(r.recall_at_k for r in results) / n if n else 0.0,
        mrr=sum(r.reciprocal_rank for r in results) / n if n else 0.0,
        by_area={a: sum(v) / len(v) for a, v in sorted(by_area.items())},
        results=results,
    )
