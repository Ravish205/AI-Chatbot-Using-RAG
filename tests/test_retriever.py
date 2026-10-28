"""
Unit tests for vector storage, similarity search, hybrid retrieval, and threshold filtering.
Compatible with both pytest and python -m unittest.
"""

import tempfile
import unittest
from src.embeddings import DeterministicMockEmbeddings
from src.retriever import Retriever
from src.text_splitter import DocumentChunk
from src.vector_store import VectorStoreManager


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_embeddings = DeterministicMockEmbeddings(dim=64)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vector_store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=self.temp_dir.name,
            embedding_provider=self.mock_embeddings,
        )
        self.sample_chunks = [
            DocumentChunk(
                page_content="Python 3.11 introduces task groups in asyncio and specialized adaptive interpreters.",
                metadata={
                    "source_filename": "python_release.txt",
                    "file_hash": "hash_py_311",
                    "page_number": 1,
                    "file_type": ".txt",
                },
                chunk_id="chunk_1",
            ),
            DocumentChunk(
                page_content="ChromaDB is an open-source embedding database designed for AI application development.",
                metadata={
                    "source_filename": "chroma_overview.txt",
                    "file_hash": "hash_chroma_001",
                    "page_number": 1,
                    "file_type": ".txt",
                },
                chunk_id="chunk_2",
            ),
            DocumentChunk(
                page_content="Streamlit is an open-source Python framework that turns data scripts into shareable web apps.",
                metadata={
                    "source_filename": "streamlit_doc.txt",
                    "file_hash": "hash_streamlit_01",
                    "page_number": 2,
                    "file_type": ".txt",
                },
                chunk_id="chunk_3",
            ),
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_indexing_and_deduplication(self):
        # Initial indexing
        result = self.vector_store.add_chunks(self.sample_chunks, skip_if_exists=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["added_chunks"], 3)

        # Duplicate check should detect existing file hash
        self.assertTrue(self.vector_store.is_file_indexed("hash_py_311"))
        self.assertFalse(self.vector_store.is_file_indexed("non_existent_hash"))

        # Adding same chunks again with skip_if_exists=True should skip
        dup_result = self.vector_store.add_chunks(self.sample_chunks[:1], skip_if_exists=True)
        self.assertEqual(dup_result["status"], "skipped")

    def test_retriever_top_k_and_hybrid(self):
        self.vector_store.add_chunks(self.sample_chunks)

        retriever = Retriever(
            vector_store=self.vector_store,
            top_k=2,
            similarity_threshold=0.0,
            enable_hybrid_search=True,
        )

        results = retriever.retrieve("What is ChromaDB used for?", top_k=2)
        self.assertLessEqual(len(results), 2)
        self.assertTrue(all(hasattr(r, "citation") for r in results))
        self.assertTrue(all(hasattr(r, "score") for r in results))

    def test_empty_vector_store_retrieval(self):
        retriever = Retriever(vector_store=self.vector_store, top_k=3)
        results = retriever.retrieve("Any question")
        self.assertEqual(results, [])

    def test_similarity_threshold_filtering(self):
        self.vector_store.add_chunks(self.sample_chunks)

        # Unreasonably high threshold should filter out results
        strict_retriever = Retriever(
            vector_store=self.vector_store,
            top_k=5,
            similarity_threshold=0.9999,
            enable_hybrid_search=False,
        )
        results = strict_retriever.retrieve("Irrelevant nonsense query")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
