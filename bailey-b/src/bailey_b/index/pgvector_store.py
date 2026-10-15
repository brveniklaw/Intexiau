"""Postgres + pgvector chunk store (Supabase in production, local Postgres in dev).

One table holds every chunk with its dense and sparse vectors. Requires
pgvector >= 0.7 for the ``sparsevec`` type; Supabase ships a current version.

Search (step 3) uses cosine distance on ``dense`` and inner product on
``sparse``. The dense column gets an HNSW index; the sparse column is scanned
exactly, which is fast at statute scale and avoids the HNSW limit of 1,000
non-zero entries per sparse vector.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

import psycopg
from pgvector import SparseVector
from pgvector.psycopg import register_vector

from bailey_b.index.embedder import Embedding
from bailey_b.models import Chunk

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


@dataclass(frozen=True)
class StoredState:
    text_sha256: str
    embedding_model: str


class PgVectorStore:
    def __init__(self, dsn: str, collection: str, *, dense_dim: int, sparse_dim: int):
        if not _IDENT_RE.match(collection):
            raise ValueError(f"collection must be a lowercase SQL identifier: {collection!r}")
        self.table = f"{collection}_chunks"
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self.conn = psycopg.connect(dsn, autocommit=False)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()
        register_vector(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def ensure_schema(self) -> None:
        t = self.table
        with self.conn.transaction():
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    chunk_id         text PRIMARY KEY,
                    corpus           text NOT NULL,
                    citation_id      text NOT NULL,
                    display_citation text NOT NULL,
                    title            text NOT NULL DEFAULT '',
                    text             text NOT NULL,
                    source_url       text,
                    effective_date   date,
                    retrieved_at     date,
                    metadata         jsonb NOT NULL DEFAULT '{{}}',
                    text_sha256      text NOT NULL,
                    embedding_model  text NOT NULL,
                    dense            vector({self.dense_dim}) NOT NULL,
                    sparse           sparsevec({self.sparse_dim}) NOT NULL,
                    indexed_at       timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_corpus_idx ON {t} (corpus)")
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_citation_idx ON {t} (citation_id)")
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_dense_idx ON {t} "
                "USING hnsw (dense vector_cosine_ops)"
            )

    def state(self, corpus: str) -> dict[str, StoredState]:
        rows = self.conn.execute(
            f"SELECT chunk_id, text_sha256, embedding_model FROM {self.table} WHERE corpus = %s",
            (corpus,),
        ).fetchall()
        self.conn.commit()
        return {cid: StoredState(sha, model) for cid, sha, model in rows}

    def upsert(
        self,
        corpus: str,
        items: Iterable[tuple[Chunk, str, Embedding]],
        embedding_model: str,
    ) -> int:
        """Insert or replace chunks with fresh vectors. ``items`` = (chunk, text_sha256, emb)."""
        rows = [
            (
                c.chunk_id,
                corpus,
                c.citation_id,
                c.display_citation,
                c.title,
                c.text,
                c.source_url,
                c.effective_date,
                c.retrieved_at,
                json.dumps(c.metadata),
                sha,
                embedding_model,
                e.dense,
                SparseVector(e.sparse, self.sparse_dim),
            )
            for c, sha, e in items
        ]
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table} (chunk_id, corpus, citation_id, display_citation, title,
                    text, source_url, effective_date, retrieved_at, metadata, text_sha256,
                    embedding_model, dense, sparse)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    corpus = EXCLUDED.corpus, citation_id = EXCLUDED.citation_id,
                    display_citation = EXCLUDED.display_citation, title = EXCLUDED.title,
                    text = EXCLUDED.text, source_url = EXCLUDED.source_url,
                    effective_date = EXCLUDED.effective_date,
                    retrieved_at = EXCLUDED.retrieved_at, metadata = EXCLUDED.metadata,
                    text_sha256 = EXCLUDED.text_sha256,
                    embedding_model = EXCLUDED.embedding_model,
                    dense = EXCLUDED.dense, sparse = EXCLUDED.sparse, indexed_at = now()
                """,
                rows,
            )
        return len(rows)

    def update_metadata(self, corpus: str, chunks: Iterable[Chunk]) -> int:
        """Refresh non-vector fields of chunks whose text (and so vectors) did not change."""
        rows = [
            (
                corpus,
                c.citation_id,
                c.display_citation,
                c.title,
                c.source_url,
                c.effective_date,
                c.retrieved_at,
                json.dumps(c.metadata),
                c.chunk_id,
            )
            for c in chunks
        ]
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(
                f"""
                UPDATE {self.table} SET corpus = %s, citation_id = %s, display_citation = %s,
                    title = %s, source_url = %s, effective_date = %s, retrieved_at = %s,
                    metadata = %s
                WHERE chunk_id = %s
                """,
                rows,
            )
        return len(rows)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        with self.conn.transaction():
            cur = self.conn.execute(f"DELETE FROM {self.table} WHERE chunk_id = ANY(%s)", (ids,))
        return cur.rowcount

    def counts(self) -> dict[str, tuple[int, int]]:
        """``{corpus: (chunks, distinct citations)}``."""
        rows = self.conn.execute(
            f"SELECT corpus, count(*), count(DISTINCT citation_id) FROM {self.table} "
            "GROUP BY corpus ORDER BY corpus"
        ).fetchall()
        self.conn.commit()
        return {corpus: (n, cites) for corpus, n, cites in rows}

    def drop(self) -> None:
        with self.conn.transaction():
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    # --- Search (step 3) -------------------------------------------------------

    _CHUNK_COLUMNS = (
        "chunk_id, citation_id, display_citation, title, text, source_url, "
        "effective_date, retrieved_at, metadata"
    )

    @staticmethod
    def _chunk(row: tuple) -> Chunk:
        cid, cite, display, title, text, url, eff, ret, meta = row
        return Chunk(
            chunk_id=cid,
            citation_id=cite,
            display_citation=display,
            title=title,
            text=text,
            source_url=url,
            effective_date=eff,
            retrieved_at=ret,
            metadata=meta or {},
        )

    def search_dense(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        """Nearest chunks by meaning: ``(chunk, cosine similarity)``, best first."""
        rows = self.conn.execute(
            f"SELECT {self._CHUNK_COLUMNS}, 1 - (dense <=> %s::vector) FROM {self.table} "
            "ORDER BY dense <=> %s::vector LIMIT %s",
            (vector, vector, limit),
        ).fetchall()
        self.conn.commit()
        return [(self._chunk(r[:-1]), float(r[-1])) for r in rows]

    def search_sparse(self, weights: dict[int, float], limit: int) -> list[tuple[Chunk, float]]:
        """Best keyword matches: ``(chunk, inner product of token weights)``, best first."""
        if not weights:
            return []
        query = SparseVector(weights, self.sparse_dim)
        rows = self.conn.execute(
            f"SELECT {self._CHUNK_COLUMNS}, -(sparse <#> %s) FROM {self.table} "
            "WHERE -(sparse <#> %s) > 0 ORDER BY sparse <#> %s LIMIT %s",
            (query, query, query, limit),
        ).fetchall()
        self.conn.commit()
        return [(self._chunk(r[:-1]), float(r[-1])) for r in rows]

    def citation_ids(self) -> set[str]:
        """Every citation in the store; the closed world that verification (step 4) checks."""
        rows = self.conn.execute(f"SELECT DISTINCT citation_id FROM {self.table}").fetchall()
        self.conn.commit()
        return {r[0] for r in rows}
