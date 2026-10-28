"""
Unit tests for recursive text chunking and metadata preservation.
Compatible with both pytest and python -m unittest.
"""

import unittest
from src.document_loader import LoadedDocument
from src.text_splitter import DocumentChunk, TextSplitter


class TestTextSplitter(unittest.TestCase):

    def test_invalid_overlap_raises_value_error(self):
        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=500, chunk_overlap=500)

        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=500, chunk_overlap=600)

    def test_chunking_creates_valid_splits(self):
        splitter = TextSplitter(chunk_size=200, chunk_overlap=40)
        long_text = (
            "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLM responses. "
            "It combines parametric knowledge from model weights with non-parametric knowledge from external sources. "
            "By indexing documents into a vector database like ChromaDB, relevant context is retrieved at inference time. "
            "This significantly decreases hallucinations and provides clear source citations. "
            "In enterprise environments, RAG ensures data freshness and compliance with confidentiality guidelines."
        )

        doc = LoadedDocument(
            page_content=long_text,
            metadata={
                "source_filename": "architecture.txt",
                "file_hash": "a1b2c3d4e5f67890",
                "page_number": 1,
            },
        )

        chunks = splitter.split_documents([doc])
        self.assertGreater(len(chunks), 1)

        for idx, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertLessEqual(len(chunk.page_content), 260)  # Allowing separator margins
            self.assertEqual(chunk.metadata["source_filename"], "architecture.txt")
            self.assertEqual(chunk.metadata["chunk_index"], idx)
            self.assertEqual(chunk.metadata["total_chunks"], len(chunks))
            self.assertIn("chunk_id", chunk.metadata)
            self.assertEqual(chunk.metadata["chunk_char_length"], len(chunk.page_content))

    def test_empty_documents_handling(self):
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([])
        self.assertEqual(chunks, [])

        doc = LoadedDocument(page_content="   ", metadata={})
        chunks = splitter.split_documents([doc])
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
