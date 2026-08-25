import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None


class Settings(BaseSettings):
    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

    # API Configuration
    openai_api_key: str = "mock-key"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = Path("data/raw")
    data_vault_dir: Path = Path("data/vault")
    kuzu_db_path: Path = Path("db_kuzu")
    chroma_db_path: Path = Path("db_chroma")

    # Pipeline & Retrieval Settings
    vector_top_k: int = 5
    graph_hops: int = 2
    rerank_top_k: int = 3
    fuzzy_dedup_threshold: float = 0.85

    def resolve_paths(self):
        """Ensure paths are resolved relative to base directory if relative."""
        if not self.data_raw_dir.is_absolute():
            self.data_raw_dir = self.base_dir / self.data_raw_dir
        if not self.data_vault_dir.is_absolute():
            self.data_vault_dir = self.base_dir / self.data_vault_dir
        if not self.kuzu_db_path.is_absolute():
            self.kuzu_db_path = self.base_dir / self.kuzu_db_path
        if not self.chroma_db_path.is_absolute():
            self.chroma_db_path = self.base_dir / self.chroma_db_path

        # Create directories if they don't exist
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_vault_dir.mkdir(parents=True, exist_ok=True)
        self.kuzu_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.resolve_paths()
