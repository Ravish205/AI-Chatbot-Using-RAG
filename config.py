"""
Configuration module for the AI Chatbot Using RAG.
Loads settings from environment variables or .env file with Pydantic validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator

    class Settings(BaseSettings):
        """Application configuration settings."""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # LLM Provider Configuration
        llm_provider: str = Field(default="ollama", description="LLM provider: 'ollama' or 'openai'")
        ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
        ollama_model: str = Field(default="llama3.2", description="Ollama LLM model name")
        ollama_request_timeout: float = Field(default=120.0, description="Ollama timeout in seconds")

        openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
        openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
        openai_temperature: float = Field(default=0.1, description="LLM sampling temperature")

        # Embedding Configuration
        embedding_provider: str = Field(
            default="huggingface",
            description="Embedding provider: 'huggingface', 'ollama', or 'openai'",
        )
        huggingface_embedding_model: str = Field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="Local HuggingFace embedding model",
        )
        ollama_embedding_model: str = Field(
            default="nomic-embed-text",
            description="Ollama embedding model",
        )
        openai_embedding_model: str = Field(
            default="text-embedding-3-small",
            description="OpenAI embedding model",
        )

        # ChromaDB Storage
        chroma_persist_directory: str = Field(
            default="./chroma_db",
            description="Directory for persistent vector store",
        )
        default_collection_name: str = Field(
            default="rag_knowledge_base",
            description="Default Chroma collection name",
        )

        # Chunking Settings
        chunk_size: int = Field(default=800, description="Chunk size in characters")
        chunk_overlap: int = Field(default=150, description="Chunk overlap in characters")

        # Retrieval Settings
        top_k: int = Field(default=4, description="Number of chunks to retrieve")
        similarity_threshold: float = Field(
            default=0.35,
            description="Minimum similarity score threshold (0.0 disables)",
        )
        enable_hybrid_search: bool = Field(
            default=True,
            description="Combine vector search with sparse BM25 search",
        )
        vector_weight: float = Field(default=0.7, description="Weight for vector similarity in hybrid search")
        bm25_weight: float = Field(default=0.3, description="Weight for BM25 score in hybrid search")

        # Document Upload & Security
        max_upload_size_mb: int = Field(default=25, description="Max upload file size in megabytes")
        allowed_extensions: List[str] = Field(
            default_factory=lambda: [".pdf", ".txt", ".docx"],
            description="Allowed file extensions",
        )

        # Logging
        log_level: str = Field(default="INFO", description="Logging level")

        @field_validator("allowed_extensions", mode="before")
        @classmethod
        def parse_allowed_extensions(cls, v):
            if isinstance(v, str):
                return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
            return v

        @field_validator("chunk_overlap")
        @classmethod
        def validate_overlap(cls, v, info):
            chunk_size = info.data.get("chunk_size", 800)
            if v >= chunk_size:
                raise ValueError("chunk_overlap must be strictly less than chunk_size")
            return v

    settings = Settings()

except ImportError:
    # Graceful fallback when pydantic-settings is not yet installed in the environment
    from dataclasses import dataclass, field

    # Attempt to load .env if python-dotenv is installed, or parse manually
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            except Exception:
                pass

    @dataclass
    class FallbackSettings:
        llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
        ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_request_timeout: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))

        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

        embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
        huggingface_embedding_model: str = os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        default_collection_name: str = os.getenv("DEFAULT_COLLECTION_NAME", "rag_knowledge_base")

        chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

        top_k: int = int(os.getenv("TOP_K", "4"))
        similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
        enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in ("true", "1", "yes")
        vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
        bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))

        max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
        allowed_extensions: List[str] = field(
            default_factory=lambda: [
                ext.strip().lower()
                for ext in os.getenv("ALLOWED_EXTENSIONS", ".pdf,.txt,.docx").split(",")
                if ext.strip()
            ]
        )
        log_level: str = os.getenv("LOG_LEVEL", "INFO")

    settings = FallbackSettings()


def get_persist_dir() -> Path:
    """Returns absolute path to the chroma persist directory, creating it if needed."""
    path = Path(settings.chroma_persist_directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
