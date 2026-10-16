"""Runtime configuration.

Every setting is read from the environment (prefix ``BAILEY_``) or from a
``bailey-b/.env`` file. See ``.env.example`` for the full list.
Nested settings use a double underscore, e.g. ``BAILEY_SEARCH__TOP_K=20``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The bailey-b/ directory. Relative paths in settings resolve against it, not the
# current working directory, so data never lands in the public website tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VectorBackend(StrEnum):
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class CorpusSettings(BaseModel):
    data_dir: Path = Field(default=Path("data"), validate_default=True)
    # Florida Statutes edition year to pull from Online Sunshine (flsenate.gov).
    statutes_year: int = 2026
    # Chapters in BALS practice areas; statutes ingestion is limited to these first.
    # fmt: off
    statute_chapters: list[str] = Field(
        default_factory=lambda: [
            # Family (741: domestic violence injunctions)
            "39", "61", "741", "742", "743",
            # Eviction / landlord-tenant
            "83",
            # Consumer
            "501", "559", "721",
            # Foreclosure
            "702",
            # Probate / guardianship
            "731", "732", "733", "734", "735", "736", "744",
            # Procedure across practice areas: service of process (48),
            # limitations of actions (95, needed for deadline tracking)
            "48", "95",
        ]
    )
    # fmt: on

    @field_validator("data_dir")
    @classmethod
    def _anchor_data_dir(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v


class CourtListenerSettings(BaseModel):
    base_url: str = "https://www.courtlistener.com/api/rest/v4"
    api_token: SecretStr | None = None
    # LSO Group Membership limit; the client must stay under this.
    daily_call_limit: int = 300


class LegiScanSettings(BaseModel):
    base_url: str = "https://api.legiscan.com/"
    api_key: SecretStr | None = None
    # Free tier limit.
    monthly_query_limit: int = 10_000


class EmbedSettings(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16
    use_fp16: bool = False


class VectorStoreSettings(BaseModel):
    # Production is Supabase (managed Postgres + pgvector); Qdrant kept as a local alternative.
    backend: VectorBackend = VectorBackend.PGVECTOR
    collection: str = "bailey_b"
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: SecretStr = SecretStr("postgresql://bailey:bailey@localhost:5432/bailey_b")


class SearchSettings(BaseModel):
    # Query path (build doc section 3): hybrid search -> top_k candidates -> rerank -> top_n.
    top_k: int = 20
    top_n: int = 5
    # Chunks fetched from each of dense and sparse search before fusion.
    pool: int = 40
    # Reciprocal rank fusion constant (the standard 60): higher flattens rank differences.
    rrf_k: int = 60
    # Keyword search's vote in fusion relative to meaning search (1.0 = equal).
    # 0.0 for plain-language questions: on the Oct 2026 grid (55 questions) keyword
    # votes only pushed good meaning-search candidates down. Questions that contain
    # a statute citation use sparse_weight_with_citation, where exact matching matters.
    sparse_weight: float = 0.0
    sparse_weight_with_citation: float = 1.0
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Tokens of question + section the reranker reads. Longer sections are truncated.
    rerank_max_tokens: int = 1024
    rerank: bool = True
    # Final order = rerank_weight * reranker score + (1 - rerank_weight) * fusion-rank
    # score (1.0 for the top candidate down to near 0 for the last). 1.0 = reranker only.
    # 0.75 was best on the Oct 2026 grid (48/55 vs 45/55 for reranker-only).
    rerank_weight: float = 0.75


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BAILEY_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    corpus: CorpusSettings = CorpusSettings()
    courtlistener: CourtListenerSettings = CourtListenerSettings()
    legiscan: LegiScanSettings = LegiScanSettings()
    embed: EmbedSettings = EmbedSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    search: SearchSettings = SearchSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
