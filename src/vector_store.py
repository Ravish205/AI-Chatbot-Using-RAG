"""
Vector store management using ChromaDB for persistent embedding storage,
document indexing, collection lifecycle management, and deduplication.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import get_persist_dir, settings
from src.embeddings import BaseEmbeddings, EmbeddingFactory
from src.text_splitter import DocumentChunk
from src.utils import get_logger

logger = get_logger(__name__)


class InMemoryVectorStore:
    """
    Lightweight in-memory vector store for offline tests and minimal environments
    when ChromaDB or native binary bindings are unavailable.
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings: List[List[float]] = []

    def count(self) -> int:
        return len(self.documents)

    def add(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: List[List[float]],
    ):
        for doc, meta, doc_id, emb in zip(documents, metadatas, ids, embeddings):
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.documents[idx] = doc
                self.metadatas[idx] = meta
                self.embeddings[idx] = emb
            else:
                self.documents.append(doc)
                self.metadatas.append(meta)
                self.ids.append(doc_id)
                self.embeddings.append(emb)

    def query_similarity(
        self,
        query_vector: List[float],
        top_k: int = 4,
    ) -> List[Tuple[str, Dict[str, Any], float, str]]:
        """Calculates cosine similarity against stored vectors."""
        if not self.documents:
            return []

        scored = []
        for doc, meta, doc_id, emb in zip(self.documents, self.metadatas, self.ids, self.embeddings):
            # Cosine similarity: dot product if normalized
            dot = sum(a * b for a, b in zip(query_vector, emb))
            norm_q = math.sqrt(sum(a * a for a in query_vector)) or 1.0
            norm_e = math.sqrt(sum(b * b for b in emb)) or 1.0
            sim = dot / (norm_q * norm_e)
            scored.append((doc, meta, float(sim), doc_id))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def delete_all(self):
        self.documents.clear()
        self.metadatas.clear()
        self.ids.clear()
        self.embeddings.clear()

    def get_indexed_files(self) -> List[str]:
        files = {m.get("source_filename") for m in self.metadatas if m.get("source_filename")}
        return sorted(list(files))

    def has_file_hash(self, file_hash: str) -> bool:
        return any(m.get("file_hash") == file_hash for m in self.metadatas)


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector storage, collection creation,
    chunk ingestion, duplicate checking, and vector similarity querying.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Union[str, Path]] = None,
        embedding_provider: Optional[BaseEmbeddings] = None,
    ):
        self.collection_name = collection_name or settings.default_collection_name
        self.persist_directory = Path(persist_directory or get_persist_dir()).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embedding_provider or EmbeddingFactory.get_embeddings()

        self._client = None
        self._collection = None
        self._fallback_store: Optional[InMemoryVectorStore] = None

        self._init_client()

    def _init_client(self):
        """Initializes persistent Chroma client or fallback in-memory store."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Initialized ChromaDB collection '%s' at '%s' (Existing chunks: %d)",
                self.collection_name,
                self.persist_directory,
                self._collection.count(),
            )
        except Exception as e:
            logger.warning(
                "ChromaDB initialization failed or module unavailable (%s). "
                "Activating lightweight in-memory vector store.",
                e,
            )
            self._fallback_store = InMemoryVectorStore(self.collection_name)

    def is_file_indexed(self, file_hash: str) -> bool:
        """Checks if a file with the given SHA-256 hash is already in the vector database."""
        if self._fallback_store is not None:
            return self._fallback_store.has_file_hash(file_hash)

        if self._collection is None:
            return False

        try:
            results = self._collection.get(
                where={"file_hash": file_hash},
                limit=1,
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.warning("Error checking file index status: %s", e)
            return False

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        skip_if_exists: bool = True,
    ) -> Dict[str, Any]:
        """
        Embeds and indexes document chunks into the vector store.
        
        Args:
            chunks: List of DocumentChunk instances.
            skip_if_exists: If True, avoids re-indexing if file hash already exists.
            
        Returns:
            Dictionary with indexing metrics (added_chunks, skipped, file_name).
        """
        if not chunks:
            return {"status": "empty", "added_chunks": 0, "message": "No chunks provided"}

        sample_meta = chunks[0].metadata
        file_hash = sample_meta.get("file_hash", "")
        file_name = sample_meta.get("source_filename", "unknown")

        if skip_if_exists and file_hash and self.is_file_indexed(file_hash):
            logger.info("Document '%s' (hash: %s...) already indexed. Skipping.", file_name, file_hash[:8])
            return {
                "status": "skipped",
                "added_chunks": 0,
                "file_name": file_name,
                "message": f"Document '{file_name}' is already indexed in the vector store.",
            }

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Generate embeddings
        logger.info("Computing embeddings for %d chunks of '%s'...", len(chunks), file_name)
        embeddings = self.embeddings.embed_documents(texts)

        if self._fallback_store is not None:
            self._fallback_store.add(texts, metadatas, ids, embeddings)
            count = len(chunks)
        else:
            # Chroma expects string/int/float/bool in metadatas, convert any complex objects
            clean_metas = []
            for m in metadatas:
                clean_m = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_m[k] = v
                    else:
                        clean_m[k] = str(v)
                clean_metas.append(clean_m)

            self._collection.upsert(
                documents=texts,
                metadatas=clean_metas,
                ids=ids,
                embeddings=embeddings,
            )
            count = len(chunks)

        logger.info("Successfully indexed %d chunks for document '%s'", count, file_name)
        return {
            "status": "success",
            "added_chunks": count,
            "file_name": file_name,
            "message": f"Successfully indexed {count} chunks for '{file_name}'.",
        }

    def similarity_search_with_score(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[str, Dict[str, Any], float, str]]:
        """
        Executes vector similarity search against the collection.
        
        Args:
            query: User search query string.
            top_k: Maximum number of chunks to return.
            
        Returns:
            List of tuples: (chunk_text, metadata, cosine_similarity_score, chunk_id).
            Similarity scores are normalized in [0.0, 1.0] where 1.0 is highest relevance.
        """
        query_vector = self.embeddings.embed_query(query)

        if self._fallback_store is not None:
            return self._fallback_store.query_similarity(query_vector, top_k=top_k)

        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            hits: List[Tuple[str, Dict[str, Any], float, str]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                # Chroma cosine distance = 1 - cosine_similarity (range [0, 2])
                # Convert distance to similarity score in [0.0, 1.0]
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                hits.append((doc, meta, similarity, cid))

            return hits
        except Exception as e:
            logger.error("Vector search query failed: %s", e)
            return []

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed chunks and metadata for BM25 indexing or inspection."""
        if self._fallback_store is not None:
            return [
                {"document": doc, "metadata": meta, "id": cid}
                for doc, meta, cid in zip(
                    self._fallback_store.documents,
                    self._fallback_store.metadatas,
                    self._fallback_store.ids,
                )
            ]

        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            results = self._collection.get(include=["documents", "metadatas"])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            ids = results.get("ids", [])
            return [
                {"document": doc, "metadata": meta, "id": cid}
                for doc, meta, cid in zip(docs, metas, ids)
            ]
        except Exception as e:
            logger.error("Failed to fetch collection chunks: %s", e)
            return []

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns aggregated summary information of all currently indexed documents."""
        all_chunks = self.get_all_chunks()
        doc_map: Dict[str, Dict[str, Any]] = {}

        for item in all_chunks:
            meta = item.get("metadata", {})
            filename = meta.get("source_filename", "Unknown")
            if filename not in doc_map:
                doc_map[filename] = {
                    "filename": filename,
                    "file_type": meta.get("file_type", ""),
                    "file_size": meta.get("file_size", 0),
                    "file_hash": meta.get("file_hash", ""),
                    "chunk_count": 0,
                    "total_characters": 0,
                }
            doc_map[filename]["chunk_count"] += 1
            doc_map[filename]["total_characters"] += len(item.get("document", ""))

        return list(doc_map.values())

    def reset_collection(self) -> None:
        """Deletes all chunks and clears the current collection."""
        if self._fallback_store is not None:
            self._fallback_store.delete_all()
            logger.info("Cleared in-memory vector store")
            return

        if self._client and self._collection:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Reset collection '%s'", self.collection_name)
            except Exception as e:
                logger.error("Failed to reset collection '%s': %s", self.collection_name, e)

    def count(self) -> int:
        """Returns total number of chunks indexed in the collection."""
        if self._fallback_store is not None:
            return self._fallback_store.count()
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return 0
        return 0
