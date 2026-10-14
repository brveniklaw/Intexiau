"""Runtime configuration.

Every setting is read from the environment (prefix ``BAILEY_``) or from a
``.env`` file in the working directory. See ``.env.example`` for the full list.
Nested settings use a double underscore, e.g. ``BAILEY_SEARCH__TOP_K=20``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorBackend(StrEnum):
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class CorpusSettings(BaseModel):
    data_dir: Path = Path("data")
    # Florida Statutes edition year to pull from Online Sunshine (flsenate.gov).
    statutes_year: int = 2026
    # Chapters in BALS practice areas; statutes ingestion is limited to these first.
    # fmt: off
    statute_chapters: list[str] = Field(
        default_factory=lambda: [
            # Family
            "39", "61", "742", "743",
            # Eviction / landlord-tenant
            "83",
            # Consumer
            "501", "559", "721",
            # Foreclosure
            "702",
            # Probate / guardianship
            "731", "732", "733", "734", "735", "736", "744",
        ]
    )
    # fmt: on


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
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BAILEY_",
        env_nested_delimiter="__",
        env_file=".env",
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
