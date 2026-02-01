"""Document processing service for loading and chunking documents."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pika.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FileTooLargeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""

    def __init__(self, filename: str, size_mb: float, max_mb: int):
        self.filename = filename
        self.size_mb = size_mb
        self.max_mb = max_mb
        super().__init__(
            f"File '{filename}' is {size_mb:.1f}MB, exceeds maximum of {max_mb}MB"
        )


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""

    content: str
    source: str
    chunk_index: int
    total_chunks: int


@dataclass
class DocumentInfo:
    """Information about a document file."""

    filename: str
    path: str
    size_bytes: int
    modified_at: datetime
    file_type: str


class DocumentProcessor:
    """Process and chunk documents from various formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.documents_dir = Path(self.settings.documents_dir)
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap

    def _ensure_documents_dir(self) -> None:
        """Create documents directory if it doesn't exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[DocumentInfo]:
        """List all supported documents in the documents directory."""
        self._ensure_documents_dir()
        documents = []

        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                documents.append(
                    DocumentInfo(
                        filename=file_path.name,
                        path=str(file_path),
                        size_bytes=stat.st_size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime),
                        file_type=file_path.suffix.lower().lstrip("."),
                    )
                )

        return sorted(documents, key=lambda d: d.filename)

    def extract_text(self, file_path: Path | str) -> str:
        """Extract text content from a document."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix in {".txt", ".md"}:
            return self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from a DOCX file."""
        from docx import Document

        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)

    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from a plain text or markdown file."""
        return file_path.read_text(encoding="utf-8")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation followed by whitespace
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)
        # Clean up whitespace
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split text into chunks using sentence-based splitting."""
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size and we have content
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Calculate overlap - keep sentences from end that fit in overlap
                overlap_chunk: list[str] = []
                overlap_length = 0

                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_document(self, file_path: Path | str) -> list[DocumentChunk]:
        """Process a document and return chunks with metadata.

        Raises:
            FileTooLargeError: If file exceeds max_upload_size_mb setting.
        """
        file_path = Path(file_path)

        # Check file size before processing (safety net for files not uploaded via API)
        max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            raise FileTooLargeError(
                file_path.name, size_mb, self.settings.max_upload_size_mb
            )

        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)

        return [
            DocumentChunk(
                content=chunk,
                source=file_path.name,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            for i, chunk in enumerate(chunks)
        ]

    def process_all_documents(self) -> list[DocumentChunk]:
        """Process all documents in the documents directory."""
        all_chunks = []

        for doc_info in self.list_documents():
            try:
                chunks = self.process_document(doc_info.path)
                all_chunks.extend(chunks)
            except Exception as e:
                # Log error but continue processing other documents
                logger.warning(f"Error processing {doc_info.filename}: {e}")

        return all_chunks


def get_document_processor() -> DocumentProcessor:
    """Get document processor instance."""
    return DocumentProcessor()
