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

    # LLM & Provider Configuration
    llm_provider: str = "ollama"  # Options: "gemini", "ollama", "openai", "groq"
    openai_api_key: str = "ollama-local-key"
    openai_base_url: str = "http://localhost:11434/v1"
    ollama_api_base: str = "http://localhost:11434"
    llm_model: str = "ollama/gemma:2b"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ollama Hardware & Execution Optimization (RTX 2050 / Local)
    ollama_num_gpu: int = 99
    ollama_num_thread: int = 4
    ollama_num_ctx: int = 2048
    ollama_num_predict: int = 512


    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = Path("data/raw")
    data_vault_dir: Path = Path("data/vault")
    data_markdown_dir: Path = Path("data/markdown")
    data_cache_dir: Path = Path("data/cache")
    kuzu_db_path: Path = Path("db_kuzu")
    chroma_db_path: Path = Path("db_chroma")

    # Pipeline & Retrieval Settings
    vector_top_k: int = 5
    graph_hops: int = 2
    rerank_top_k: int = 3
    rerank_min_score: float = 0.0  # cross-encoder logit floor: weaker candidates are noise, not context
    fuzzy_dedup_threshold: float = 0.85

    def resolve_paths(self):
        """Ensure paths are resolved relative to base directory if relative."""
        if not self.data_raw_dir.is_absolute():
            self.data_raw_dir = self.base_dir / self.data_raw_dir
        if not self.data_vault_dir.is_absolute():
            self.data_vault_dir = self.base_dir / self.data_vault_dir
        if not self.data_markdown_dir.is_absolute():
            self.data_markdown_dir = self.base_dir / self.data_markdown_dir
        if not self.data_cache_dir.is_absolute():
            self.data_cache_dir = self.base_dir / self.data_cache_dir
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
