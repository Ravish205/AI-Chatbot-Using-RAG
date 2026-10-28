"""
Text splitter module for intelligent recursive chunking with metadata preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from src.document_loader import LoadedDocument
from src.utils import calculate_text_hash, get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents an individual text chunk ready for vector embedding and retrieval."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def to_langchain_doc(self):
        """Converts chunk to LangChain Document representation."""
        try:
            from langchain_core.documents import Document
            return Document(page_content=self.page_content, metadata=self.metadata)
        except ImportError:
            return self


class TextSplitter:
    """
    Splits loaded documents into overlapping chunks using recursive character boundaries
    while preserving comprehensive source metadata and tracking chunk indices.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )

    def split_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """
        Splits a list of LoadedDocument instances into DocumentChunks.
        
        Args:
            documents: List of LoadedDocument objects.
            
        Returns:
            List of DocumentChunk objects with unique IDs and updated metadata.
        """
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue

            base_metadata = dict(doc.metadata)
            raw_splits = self._recursive_split(text, self.chunk_size, self.chunk_overlap)

            total_chunks = len(raw_splits)
            doc_hash = base_metadata.get("file_hash", "doc")
            page_num = base_metadata.get("page_number", 1)

            for idx, split_text in enumerate(raw_splits):
                chunk_hash = calculate_text_hash(split_text)[:10]
                chunk_id = f"{doc_hash[:12]}_p{page_num}_c{idx}_{chunk_hash}"

                chunk_meta = {
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_char_length": len(split_text),
                }

                all_chunks.append(
                    DocumentChunk(
                        page_content=split_text,
                        metadata=chunk_meta,
                        chunk_id=chunk_id,
                    )
                )

        logger.info(
            "Split %d document page(s) into %d text chunks (chunk_size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Recursively splits text using hierarchical separators to preserve semantic structure.
        """
        # Try using LangChain's splitter if available
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
            return splitter.split_text(text)
        except ImportError:
            pass

        # Standalone pure Python recursive splitter
        return self._pure_python_split(text, chunk_size, chunk_overlap, self.separators)

    def _pure_python_split(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
    ) -> List[str]:
        """Fallback recursive splitting algorithm in pure Python."""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chosen_sep = separators[-1]
        for sep in separators:
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                break

        splits = text.split(chosen_sep) if chosen_sep else list(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for segment in splits:
            seg_len = len(segment) + (len(chosen_sep) if current_chunk else 0)
            if current_len + seg_len > chunk_size and current_chunk:
                joined = (chosen_sep.join(current_chunk)).strip()
                if joined:
                    chunks.append(joined)

                # Keep overlap from the end of current_chunk
                overlap_tokens: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= chunk_overlap:
                        overlap_tokens.insert(0, item)
                        overlap_len += len(item) + len(chosen_sep)
                    else:
                        break
                current_chunk = overlap_tokens
                current_len = sum(len(x) for x in current_chunk) + max(0, len(current_chunk) - 1) * len(chosen_sep)

            current_chunk.append(segment)
            current_len += len(segment) + (len(chosen_sep) if len(current_chunk) > 1 else 0)

        if current_chunk:
            final_joined = (chosen_sep.join(current_chunk)).strip()
            if final_joined:
                chunks.append(final_joined)

        return chunks
