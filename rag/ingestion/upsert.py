import time
from typing import List, Optional, Any, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    PointStruct,
    Distance,
    CollectionInfo
)
from qdrant_client.http.exceptions import UnexpectedResponse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.chunk import Chunk
from rag.ingestion.embedder import Embedder


class QdrantUpsertError(Exception):
    """Custom exception raised when Qdrant upsert operations fail."""
    pass


class QdrantUpserter:
    """
    Production-ready Qdrant vector database upserter.

    Features:
    - Automatic collection creation with proper configuration
    - Batched upsert operations (200 points per batch)
    - Retry logic for transient failures
    - JSON-serializable payload validation
    - Comprehensive error handling
    """

    DEFAULT_BATCH_SIZE = 200
    DEFAULT_VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension
    MAX_RETRIES = 1

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None
    ):
        """
        Initialize Qdrant upserter.

        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant server URL
            api_key: Optional API key for authentication
            vector_size: Embedding dimension (auto-detected if not provided)

        Raises:
            QdrantUpsertError: If connection to Qdrant fails
        """
        self.collection_name = collection_name
        self.url = url
        self.vector_size = vector_size or self.DEFAULT_VECTOR_SIZE

        try:
            # Initialize Qdrant client
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=30
            )

            # Verify connection
            self.client.get_collections()

        except Exception as e:
            raise QdrantUpsertError(
                f"Failed to connect to Qdrant at {url}: {str(e)}"
            )

    def ensure_collection(self, vector_size: Optional[int] = None):
        """
        Ensure collection exists, create if it doesn't.

        Args:
            vector_size: Override default vector size
        """
        if vector_size:
            self.vector_size = vector_size

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name} "
                      f"(dimension={self.vector_size}, distance=cosine)")
            else:
                # Verify vector size matches
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                if existing_size != self.vector_size:
                    raise QdrantUpsertError(
                        f"Collection '{self.collection_name}' exists with vector size {existing_size}, "
                        f"but embedder produces size {self.vector_size}. "
                        f"Delete the collection or use matching vector size."
                    )

        except QdrantUpsertError:
            raise
        except Exception as e:
            raise QdrantUpsertError(f"Failed to ensure collection exists: {str(e)}")

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Upsert chunks into Qdrant in batches.

        Args:
            chunks: List of Chunk objects with embeddings
            batch_size: Number of points to upsert per batch

        Raises:
            QdrantUpsertError: If upsert operation fails after retries
        """
        if not chunks:
            return

        # Validate chunks have embeddings
        for chunk in chunks:
            if not chunk.embedding_vector:
                raise QdrantUpsertError(
                    f"Chunk {chunk.id} has no embedding vector. "
                    f"Run embedder.embed() before upserting."
                )

        # Ensure collection exists with correct dimension
        if chunks[0].embedding_vector:
            self.ensure_collection(vector_size=len(chunks[0].embedding_vector))

        # Convert chunks to Qdrant points
        points = self._chunks_to_points(chunks)

        # Upsert in batches
        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_points + batch_size - 1) // batch_size

            try:
                self._upsert_batch_with_retry(batch)
                print(f"Upserted batch {batch_num}/{total_batches} "
                      f"({len(batch)} points) to {self.collection_name}")
            except Exception as e:
                raise QdrantUpsertError(
                    f"Failed to upsert batch {batch_num}/{total_batches}: {str(e)}"
                )

        print(f"Successfully upserted {total_points} points to {self.collection_name}")

    def _chunks_to_points(self, chunks: List[Chunk]) -> List[PointStruct]:
        """
        Convert Chunk objects to Qdrant PointStruct objects.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of PointStruct objects ready for upsert
        """
        points = []

        for chunk in chunks:
            # Build payload with all chunk metadata
            payload = self._build_payload(chunk)

            # Create Qdrant point
            point = PointStruct(
                id=chunk.id,
                vector=chunk.embedding_vector,
                payload=payload
            )

            points.append(point)

        return points

    def _build_payload(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Build JSON-serializable payload from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Dictionary with all relevant chunk data
        """
        # Start with chunk metadata
        payload = dict(chunk.metadata) if chunk.metadata else {}

        # Add core chunk fields
        payload.update({
            'document_id': chunk.document_id,
            'org_id': chunk.org_id,
            'assistant_ids': chunk.assistant_ids,
            'text': chunk.text,
            'sparse_tokens': chunk.sparse_tokens if chunk.sparse_tokens else []
        })

        # Ensure all values are JSON-serializable
        payload = self._ensure_json_serializable(payload)

        return payload

    def _ensure_json_serializable(self, data: Any) -> Any:
        """
        Recursively ensure all data is JSON-serializable.

        Args:
            data: Data to validate

        Returns:
            JSON-serializable version of data
        """
        if isinstance(data, dict):
            return {k: self._ensure_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            # Convert other types to string
            return str(data)

    def _upsert_batch_with_retry(self, points: List[PointStruct]):
        """
        Upsert a batch of points with retry logic.

        Args:
            points: List of PointStruct to upsert

        Raises:
            Exception: If upsert fails after all retries
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                return  # Success

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    print(f"Upsert attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    break

        # All retries exhausted
        raise last_error


def ingest_document(
    chunks: List[Chunk],
    embedder: Embedder,
    upserter: QdrantUpserter
) -> None:
    """
    Complete document ingestion pipeline.

    Pipeline:
    1. Extract text from all chunks
    2. Generate embeddings using embedder
    3. Attach embeddings to chunk objects
    4. Upsert chunks into Qdrant

    Args:
        chunks: List of Chunk objects (without embeddings)
        embedder: Embedder instance for generating vectors
        upserter: QdrantUpserter instance for database operations

    Raises:
        QdrantUpsertError: If ingestion pipeline fails
        EmbeddingError: If embedding generation fails
    """
    if not chunks:
        print("No chunks to ingest")
        return

    try:
        # Step 1: Extract text from chunks
        texts = [chunk.text for chunk in chunks]

        # Step 2: Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedder.embed(texts)

        # Step 3: Attach embeddings to chunks
        if len(embeddings) != len(chunks):
            raise QdrantUpsertError(
                f"Embedding count mismatch: {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks"
            )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding_vector = embedding

        # Step 4: Upsert into Qdrant
        print(f"Upserting {len(chunks)} chunks to Qdrant...")
        upserter.upsert_chunks(chunks)

        print(f"Successfully ingested {len(chunks)} chunks")

    except Exception as e:
        raise QdrantUpsertError(f"Document ingestion failed: {str(e)}")
