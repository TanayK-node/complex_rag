"""
All tunables live here so no module hardcodes a magic number. Loaded from
environment variables (or a .env file) with sane defaults, per pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- models / providers (kept provider-agnostic; swap without touching pipeline code) ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-sonnet-4-6")
    llm_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", validation_alias="GEMINI_MODEL")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_min_interval_seconds: float = Field(
        default=13.0, validation_alias="GEMINI_MIN_INTERVAL_SECONDS"
    )

    # --- chunking ---
    chunk_max_tokens: int = Field(default=400)
    chunk_min_tokens: int = Field(default=40)
    chunk_overlap_tokens: int = Field(default=40)
    chunk_inherit_parent_context: bool = Field(default=True)
    chunk_include_section_title: bool = Field(default=True)

    # --- retrieval ---
    vector_top_k: int = Field(default=10)
    bm25_top_k: int = Field(default=10)
    hierarchy_top_k: int = Field(default=5)
    rerank_top_k: int = Field(default=6)
    similarity_threshold: float = Field(default=0.28)

    # --- cross-reference expansion ---
    max_cross_reference_hops: int = Field(default=1)

    # --- generation / verification ---
    max_context_tokens: int = Field(default=3000)
    groundedness_retry_limit: int = Field(default=2)

    # --- storage ---
    vector_store_path: str = Field(default="./data/processed/vector_index")
    metadata_db_path: str = Field(default="./data/processed/metadata.sqlite")

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"


settings = Settings()
