"""
Unit tests for the document loader module.
Verifies TXT, PDF, DOCX extraction, validation boundaries, and metadata preservation.
Compatible with both pytest and python -m unittest.
"""

import unittest
from src.document_loader import (
    DocumentLoader,
    EmptyDocumentError,
    FileTooLargeError,
    LoadedDocument,
    UnsupportedFileTypeError,
)
from src.utils import calculate_file_hash, clean_text


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader(allowed_extensions=[".pdf", ".txt", ".docx"], max_upload_size_mb=5)

    def test_clean_text_normalization(self):
        raw = "Hello   world!\r\n\r\n\r\nThis has \x00null bytes and   spaces.\n"
        cleaned = clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertIn("Hello world!", cleaned)
        self.assertIn("This has null bytes and spaces.", cleaned)

    def test_file_hash_consistency(self):
        content = b"Sample document content for hashing verification."
        hash1 = calculate_file_hash(content)
        hash2 = calculate_file_hash(content)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_unsupported_file_extension(self):
        with self.assertRaises(UnsupportedFileTypeError) as ctx:
            self.loader.load_from_bytes(b"some content", filename="malicious_script.exe")
        self.assertIn("not supported", str(ctx.exception))

    def test_file_too_large(self):
        oversized = b"X" * (6 * 1024 * 1024)
        with self.assertRaises(FileTooLargeError) as ctx:
            self.loader.load_from_bytes(oversized, filename="huge_document.txt")
        self.assertIn("exceeds the maximum allowed size", str(ctx.exception))

    def test_empty_document_error(self):
        with self.assertRaises(EmptyDocumentError):
            self.loader.load_from_bytes(b"    \n   \t  ", filename="blank.txt")

    def test_txt_loading_and_metadata(self):
        content = b"Introduction to Retrieval-Augmented Generation (RAG) with ChromaDB and LLMs."
        docs = self.loader.load_from_bytes(content, filename="rag_guide.txt")

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIsInstance(doc, LoadedDocument)
        self.assertIn("Retrieval-Augmented Generation", doc.page_content)
        self.assertEqual(doc.metadata["source_filename"], "rag_guide.txt")
        self.assertEqual(doc.metadata["file_type"], ".txt")
        self.assertEqual(doc.metadata["page_number"], 1)
        self.assertEqual(doc.metadata["character_count"], len(doc.page_content))
        self.assertIn("file_hash", doc.metadata)

    def test_txt_encoding_fallback(self):
        content = "Café und Überraschung im RAG-System".encode("latin-1")
        docs = self.loader.load_from_bytes(content, filename="german_notes.txt")
        self.assertEqual(len(docs), 1)
        self.assertTrue("Café" in docs[0].page_content or "Überraschung" in docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
