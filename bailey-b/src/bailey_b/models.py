"""Core data types shared by every stage of the pipeline.

These are the contract between corpus ingestion, indexing, search, citation
verification and the eval harness. Every chunk carries a canonical
``citation_id`` so that verification is a set-membership check (legal citation
is a closed-world problem: an ID is either in the corpus or it is not).

Canonical citation ID formats:

- ``fl-stat:83.56``          Florida Statutes section (subsections live in the text, not the ID)
- ``cl-opinion:2481923``     CourtListener opinion ID
- ``fl-2dca:<docket>``       Florida Second DCA opinion, by docket number
- ``fl-sc:<docket>``         Florida Supreme Court opinion, by docket number
- ``cap:<case_id>``          Caselaw Access Project case ID
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SourceKind(StrEnum):
    FL_STATUTE = "fl-stat"
    COURTLISTENER = "cl-opinion"
    FL_2DCA = "fl-2dca"
    FL_SUPREME = "fl-sc"
    CAP = "cap"


CITATION_ID_RE = re.compile(
    r"^(?P<kind>" + "|".join(re.escape(k.value) for k in SourceKind) + r"):(?P<key>\S+)$"
)


def parse_citation_id(citation_id: str) -> tuple[SourceKind, str]:
    """Split a canonical citation ID into (kind, key). Raises ValueError if malformed."""
    m = CITATION_ID_RE.match(citation_id)
    if not m:
        raise ValueError(f"malformed citation id: {citation_id!r}")
    return SourceKind(m["kind"]), m["key"]


class Chunk(BaseModel):
    """One indexable unit: a statute section, or an opinion (or part of one)."""

    chunk_id: str = Field(description="Unique within the index, e.g. 'fl-stat:83.56#0'.")
    citation_id: str = Field(description="Canonical citation this chunk belongs to.")
    display_citation: str = Field(description="Human-readable cite, e.g. 'Fla. Stat. § 83.56'.")
    title: str = ""
    text: str
    source_url: str | None = None
    # Edition year for statutes, decision date for opinions. Drives staleness checks.
    effective_date: date | None = None
    retrieved_at: date | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("citation_id")
    @classmethod
    def _valid_citation_id(cls, v: str) -> str:
        parse_citation_id(v)
        return v


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval function, with its scores."""

    chunk: Chunk
    rank: int
    score: float = Field(description="Final score (reranker score once reranking exists).")
    dense_score: float | None = None
    sparse_score: float | None = None
