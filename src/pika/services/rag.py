"""RAG engine using ChromaDB and sentence-transformers."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from pika.config import Settings, get_settings
from pika.services.documents import DocumentProcessor, get_document_processor
from pika.services.ollama import OllamaClient, get_ollama_client


class Confidence(str, Enum):
    """Confidence level for query results."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Source:
    """A source document referenced in the answer."""

    filename: str
    chunk_index: int
    content: str
    similarity: float


@dataclass
class QueryResult:
    """Result from a RAG query."""

    answer: str
    sources: list[Source]
    confidence: Confidence


@dataclass
class IndexStats:
    """Statistics about the current index."""

    total_documents: int
    total_chunks: int
    collection_name: str


class RAGEngine:
    """RAG engine with ChromaDB vector store and sentence-transformers embeddings."""

    COLLECTION_NAME = "pika_documents"

    def __init__(
        self,
        settings: Settings | None = None,
        document_processor: DocumentProcessor | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.document_processor = document_processor or get_document_processor()
        self.ollama_client = ollama_client or get_ollama_client()

        # Initialize embedding model
        self._embedding_model: SentenceTransformer | None = None

        # Initialize ChromaDB
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        return self._embedding_model

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _get_confidence(self, similarities: list[float]) -> Confidence:
        """Determine confidence level based on similarity scores."""
        if not similarities:
            return Confidence.NONE

        max_similarity = max(similarities)

        if max_similarity >= self.settings.confidence_high:
            return Confidence.HIGH
        elif max_similarity >= self.settings.confidence_medium:
            return Confidence.MEDIUM
        elif max_similarity >= self.settings.confidence_low:
            return Confidence.LOW
        else:
            return Confidence.NONE

    def index_documents(self) -> IndexStats:
        """Index all documents from the documents directory."""
        # Clear existing index
        self.clear_index()

        # Process all documents
        chunks = self.document_processor.process_all_documents()

        if not chunks:
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                collection_name=self.COLLECTION_NAME,
            )

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        seen_sources = set()

        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            })
            seen_sources.add(chunk.source)

        # Generate embeddings
        embeddings = self._embed(documents)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return IndexStats(
            total_documents=len(seen_sources),
            total_chunks=len(chunks),
            collection_name=self.COLLECTION_NAME,
        )

    def get_stats(self) -> IndexStats:
        """Get current index statistics."""
        count = self.collection.count()

        # Get unique sources
        if count > 0:
            results = self.collection.get(include=["metadatas"])
            sources = set(m["source"] for m in results["metadatas"])
            doc_count = len(sources)
        else:
            doc_count = 0

        return IndexStats(
            total_documents=doc_count,
            total_chunks=count,
            collection_name=self.COLLECTION_NAME,
        )

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        try:
            self.chroma_client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
        except ValueError:
            # Collection doesn't exist
            pass

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """Query the RAG system with a question."""
        top_k = top_k or self.settings.top_k

        # Check if index has documents
        if self.collection.count() == 0:
            return QueryResult(
                answer="No documents have been indexed yet. Please index documents first.",
                sources=[],
                confidence=Confidence.NONE,
            )

        # Generate embedding for question
        question_embedding = self._embed([question])[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert distances to similarities (cosine distance to similarity)
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]

        # Build sources
        sources = []
        for i, (doc, metadata, similarity) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                similarities,
            )
        ):
            sources.append(
                Source(
                    filename=metadata["source"],
                    chunk_index=metadata["chunk_index"],
                    content=doc,
                    similarity=similarity,
                )
            )

        # Determine confidence
        confidence = self._get_confidence(similarities)

        # Generate answer based on confidence
        if confidence == Confidence.NONE:
            answer = (
                "I couldn't find relevant information in the indexed documents "
                "to answer your question. Please try rephrasing or ensure the "
                "relevant documents are indexed."
            )
        else:
            # Build context from retrieved chunks
            context_parts = []
            for source in sources:
                context_parts.append(
                    f"[From {source.filename}]:\n{source.content}"
                )
            context = "\n\n---\n\n".join(context_parts)

            # Generate answer using Ollama
            system_prompt = (
                "You are a helpful assistant answering questions based on the provided context. "
                "Use only the information from the context to answer. "
                "If the context doesn't contain enough information, say so. "
                "Be concise and cite which document(s) your answer comes from."
            )

            prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

            answer = await self.ollama_client.generate(
                prompt=prompt,
                system=system_prompt,
            )

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )


# Singleton instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
