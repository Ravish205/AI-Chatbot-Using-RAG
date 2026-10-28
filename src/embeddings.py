"""
Embeddings module providing local HuggingFace/Sentence-Transformers,
Ollama, OpenAI, and lightweight deterministic fallback embedding providers.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Optional

from config import settings
from src.utils import get_logger

logger = get_logger(__name__)


class BaseEmbeddings:
    """Base interface for embedding models."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class DeterministicMockEmbeddings(BaseEmbeddings):
    """
    Deterministic pseudo-vector generator for fast offline unit tests
    and fallback operation when no external embedding model is loaded.
    Generates consistent normalized vectors of dimension `dim` using SHA-256.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> List[float]:
        vector: List[float] = []
        for i in range(self.dim):
            seed_str = f"{text}:{i}"
            digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
            # Map hex integer to [-1.0, 1.0]
            val = ((int(digest[:8], 16) % 10000) / 5000.0) - 1.0
            vector.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


class HuggingFaceLocalEmbeddings(BaseEmbeddings):
    """Local embedding wrapper using sentence-transformers / langchain-huggingface."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.huggingface_embedding_model
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        try:
            # First try langchain-huggingface
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._backend = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("Initialized langchain_huggingface with model: %s", self.model_name)
                return
            except ImportError:
                pass

            # Next try langchain_community
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._backend = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("Initialized langchain_community HuggingFaceEmbeddings: %s", self.model_name)
                return
            except ImportError:
                pass

            # Direct sentence-transformers
            from sentence_transformers import SentenceTransformer

            class STWrapper:
                def __init__(self, model):
                    self.model = model

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    embeddings = self.model.encode(texts, normalize_embeddings=True)
                    return embeddings.tolist()

                def embed_query(self, text: str) -> List[float]:
                    embedding = self.model.encode([text], normalize_embeddings=True)
                    return embedding[0].tolist()

            st_model = SentenceTransformer(self.model_name, device="cpu")
            self._backend = STWrapper(st_model)
            logger.info("Initialized raw SentenceTransformer: %s", self.model_name)

        except Exception as e:
            logger.warning(
                "Could not load HuggingFace model '%s' (%s). Falling back to DeterministicMockEmbeddings.",
                self.model_name,
                e,
            )
            self._backend = DeterministicMockEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._backend.embed_query(text)


class OllamaLocalEmbeddings(BaseEmbeddings):
    """Local embeddings via Ollama REST API or langchain-ollama."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        try:
            from langchain_ollama import OllamaEmbeddings
            self._backend = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
            )
            logger.info("Initialized OllamaEmbeddings (model: %s, url: %s)", self.model_name, self.base_url)
        except Exception as e:
            logger.warning("Failed to initialize OllamaEmbeddings: %s. Using Mock fallback.", e)
            self._backend = DeterministicMockEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._backend.embed_query(text)


class OpenAICloudEmbeddings(BaseEmbeddings):
    """OpenAI Cloud embeddings wrapper."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or settings.openai_embedding_model
        self.api_key = api_key or settings.openai_api_key
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAI embeddings.")
        try:
            from langchain_openai import OpenAIEmbeddings
            self._backend = OpenAIEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
            )
            logger.info("Initialized OpenAIEmbeddings: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize OpenAIEmbeddings: %s", e)
            raise


class EmbeddingFactory:
    """Factory to instantiate the appropriate embedding provider."""

    @staticmethod
    def get_embeddings(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Any:
        chosen_provider = (provider or settings.embedding_provider).lower()
        logger.info("Instantiating embedding provider: %s", chosen_provider)

        if chosen_provider == "huggingface":
            return HuggingFaceLocalEmbeddings(model_name=model_name)
        elif chosen_provider == "ollama":
            return OllamaLocalEmbeddings(model_name=model_name)
        elif chosen_provider == "openai":
            return OpenAICloudEmbeddings(model_name=model_name)
        elif chosen_provider in ("mock", "test", "deterministic"):
            return DeterministicMockEmbeddings()
        else:
            logger.warning(
                "Unknown embedding provider '%s'. Defaulting to HuggingFace local embeddings.",
                chosen_provider,
            )
            return HuggingFaceLocalEmbeddings(model_name=model_name)

