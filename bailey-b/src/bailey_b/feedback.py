"""Attorney feedback on retrieved results (roadmap 10.3: collect the signal early).

Each Useful / Not relevant click is stored with the question, the chunk shown,
its rank and score. Nothing reads it yet; it becomes reranker training data.
The question text can contain client facts, so this table lives only in the
BALS database, never in exports or the repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


@dataclass(frozen=True)
class Feedback:
    question: str
    chunk_id: str
    citation_id: str
    rank: int
    score: float
    useful: bool


class FeedbackStore:
    def __init__(self, dsn: str, collection: str):
        if not _IDENT_RE.match(collection):
            raise ValueError(f"collection must be a lowercase SQL identifier: {collection!r}")
        self.table = f"{collection}_feedback"
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id          bigserial PRIMARY KEY,
                created_at  timestamptz NOT NULL DEFAULT now(),
                question    text NOT NULL,
                chunk_id    text NOT NULL,
                citation_id text NOT NULL,
                rank        int NOT NULL,
                score       double precision NOT NULL,
                useful      boolean NOT NULL
            )
            """
        )

    def add(self, fb: Feedback) -> None:
        self.conn.execute(
            f"INSERT INTO {self.table} (question, chunk_id, citation_id, rank, score, useful) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (fb.question, fb.chunk_id, fb.citation_id, fb.rank, fb.score, fb.useful),
        )

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def drop(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")


class MemoryFeedbackStore:
    """For tests and for running without a database."""

    def __init__(self):
        self.items: list[Feedback] = []

    def add(self, fb: Feedback) -> None:
        self.items.append(fb)

    def count(self) -> int:
        return len(self.items)
