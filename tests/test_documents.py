"""Tests for document processing service."""

import tempfile
from pathlib import Path

import pytest


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    @pytest.fixture
    def processor(self, temp_dirs):
        """Create a document processor with temp directory."""
        from pika.services.documents import DocumentProcessor
        from pika.config import Settings

        settings = Settings(
            documents_dir=temp_dirs["docs"],
            chunk_size=100,
            chunk_overlap=20,
        )
        return DocumentProcessor(settings=settings)

    def test_list_documents_empty(self, processor):
        """Verify empty directory returns empty list."""
        documents = processor.list_documents()
        assert documents == []

    def test_list_documents_with_files(self, processor, temp_dirs):
        """Verify documents are listed correctly."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Create test files
        (docs_dir / "test1.txt").write_text("Test content 1")
        (docs_dir / "test2.md").write_text("# Test content 2")
        (docs_dir / "ignored.xyz").write_text("Should be ignored")

        documents = processor.list_documents()

        filenames = [d.filename for d in documents]
        assert "test1.txt" in filenames
        assert "test2.md" in filenames
        assert "ignored.xyz" not in filenames

    def test_extract_text_txt(self, processor, temp_dirs):
        """Verify text extraction from .txt files."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "test.txt"
        test_file.write_text("Hello, this is test content.")

        text = processor.extract_text(test_file)
        assert text == "Hello, this is test content."

    def test_extract_text_md(self, processor, temp_dirs):
        """Verify text extraction from .md files."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "test.md"
        test_file.write_text("# Header\n\nParagraph content.")

        text = processor.extract_text(test_file)
        assert "Header" in text
        assert "Paragraph content" in text

    def test_extract_text_unsupported(self, processor, temp_dirs):
        """Verify unsupported file types raise error."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "test.xyz"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.extract_text(test_file)

    def test_extract_text_pdf(self, processor, sample_pdf_file):
        """Verify text extraction from .pdf files."""
        text = processor.extract_text(sample_pdf_file)

        # The PDF contains "This is a test PDF document for PIKA testing."
        assert "test" in text.lower() or "PDF" in text or "PIKA" in text

    def test_extract_text_docx(self, processor, sample_docx_file):
        """Verify text extraction from .docx files."""
        text = processor.extract_text(sample_docx_file)

        assert "test" in text.lower()
        assert "DOCX" in text or "document" in text.lower()
        assert "paragraphs" in text.lower()

    def test_rejects_exe_file(self, processor, temp_dirs):
        """Verify .exe files are rejected."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "malware.exe"
        test_file.write_bytes(b"MZ\x00\x00")  # Minimal PE header

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.extract_text(test_file)

    def test_rejects_zip_file(self, processor, temp_dirs):
        """Verify .zip files are rejected."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "archive.zip"
        test_file.write_bytes(b"PK\x03\x04")  # ZIP magic bytes

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.extract_text(test_file)

    def test_rejects_script_file(self, processor, temp_dirs):
        """Verify script files are rejected."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        for ext in [".sh", ".bat", ".ps1", ".py"]:
            test_file = docs_dir / f"script{ext}"
            test_file.write_text("#!/bin/bash\necho hello")

            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.extract_text(test_file)

            test_file.unlink()

    def test_chunk_text_basic(self, processor):
        """Verify text chunking works."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = processor.chunk_text(text)

        assert len(chunks) >= 1
        # All original content should be in chunks
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Fourth" in combined

    def test_chunk_text_empty(self, processor):
        """Verify empty text returns empty list."""
        chunks = processor.chunk_text("")
        assert chunks == []

    def test_chunk_text_short(self, processor):
        """Verify short text returns single chunk."""
        text = "Short."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1

    def test_process_document(self, processor, temp_dirs):
        """Verify document processing returns chunks with metadata."""
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        test_file = docs_dir / "test.txt"
        test_file.write_text("First sentence. Second sentence. Third sentence.")

        chunks = processor.process_document(test_file)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source == "test.txt"
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks >= 1
            assert len(chunk.content) > 0


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Verify DocumentChunk can be created."""
        from pika.services.documents import DocumentChunk

        chunk = DocumentChunk(
            content="Test content",
            source="test.txt",
            chunk_index=0,
            total_chunks=5,
        )

        assert chunk.content == "Test content"
        assert chunk.source == "test.txt"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 5


class TestDocumentInfo:
    """Tests for DocumentInfo dataclass."""

    def test_document_info_creation(self):
        """Verify DocumentInfo can be created."""
        from datetime import datetime
        from pika.services.documents import DocumentInfo

        info = DocumentInfo(
            filename="test.txt",
            path="/path/to/test.txt",
            size_bytes=1024,
            modified_at=datetime.now(),
            file_type="txt",
        )

        assert info.filename == "test.txt"
        assert info.size_bytes == 1024
        assert info.file_type == "txt"
