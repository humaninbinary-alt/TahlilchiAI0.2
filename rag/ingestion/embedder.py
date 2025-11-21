from typing import List, Optional
import numpy as np


class EmbeddingError(Exception):
    """Custom exception raised when embedding generation fails."""
    pass


class Embedder:
    """
    Model-agnostic embedding provider wrapper.

    Supports multiple backends:
    - local: sentence-transformers (default, fully implemented)
    - openai: OpenAI API (placeholder)
    - cohere: Cohere API (placeholder)

    Features:
    - Batched embedding generation
    - Dimension consistency validation
    - Memory-efficient processing
    - Error handling and retries
    """

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, 384-dim model

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "local",
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Initialize embedder with specified backend.

        Args:
            model_name: Model identifier (backend-specific)
            backend: Embedding backend ("local", "openai", "cohere")
            batch_size: Number of texts to embed per batch

        Raises:
            EmbeddingError: If backend initialization fails
        """
        self.backend = backend.lower()
        self.batch_size = batch_size
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = None
        self.embedding_dim = None

        # Initialize the backend
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected embedding backend."""
        if self.backend == "local":
            self._initialize_local()
        elif self.backend == "openai":
            self._initialize_openai()
        elif self.backend == "cohere":
            self._initialize_cohere()
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Supported backends: local, openai, cohere"
            )

    def _initialize_local(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension from model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load local model '{self.model_name}': {str(e)}")

    def _initialize_openai(self):
        """Initialize OpenAI embedding API (placeholder)."""
        # Placeholder for OpenAI implementation
        try:
            import openai
            # Set default model if not specified
            if not self.model_name or self.model_name == self.DEFAULT_MODEL_NAME:
                self.model_name = "text-embedding-3-small"
            self.embedding_dim = 1536  # Default for text-embedding-3-small
            # Store API client (would need API key from config)
            # self.model = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise EmbeddingError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

    def _initialize_cohere(self):
        """Initialize Cohere embedding API (placeholder)."""
        # Placeholder for Cohere implementation
        try:
            import cohere
            # Set default model if not specified
            if not self.model_name or self.model_name == self.DEFAULT_MODEL_NAME:
                self.model_name = "embed-multilingual-v3.0"
            self.embedding_dim = 1024  # Default for multilingual model
            # Store API client (would need API key from config)
            # self.model = cohere.Client(api_key=api_key)
        except ImportError:
            raise EmbeddingError(
                "cohere is required for Cohere embeddings. "
                "Install with: pip install cohere"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Processes texts in batches to avoid memory issues.
        Ensures all embeddings have consistent dimensions.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text.strip() if text else "" for text in texts]
        if not any(valid_texts):
            raise EmbeddingError("Cannot embed empty texts")

        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i:i + self.batch_size]
                batch_embeddings = self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)

            # Validate dimension consistency
            self._validate_embeddings(all_embeddings)

            return all_embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts using the selected backend."""
        if self.backend == "local":
            return self._embed_batch_local(texts)
        elif self.backend == "openai":
            return self._embed_batch_openai(texts)
        elif self.backend == "cohere":
            return self._embed_batch_cohere(texts)
        else:
            raise EmbeddingError(f"Backend {self.backend} not implemented")

    def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using local sentence-transformers model."""
        # Generate embeddings (returns numpy array)
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better retrieval
        )

        # Convert numpy arrays to Python lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI API (placeholder)."""
        # Placeholder implementation
        raise EmbeddingError(
            "OpenAI backend not fully implemented. "
            "Use backend='local' or implement OpenAI API integration."
        )
        # Full implementation would look like:
        # response = self.model.embeddings.create(
        #     input=texts,
        #     model=self.model_name
        # )
        # return [item.embedding for item in response.data]

    def _embed_batch_cohere(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Cohere API (placeholder)."""
        # Placeholder implementation
        raise EmbeddingError(
            "Cohere backend not fully implemented. "
            "Use backend='local' or implement Cohere API integration."
        )
        # Full implementation would look like:
        # response = self.model.embed(
        #     texts=texts,
        #     model=self.model_name,
        #     input_type="search_document"
        # )
        # return response.embeddings

    def _validate_embeddings(self, embeddings: List[List[float]]):
        """Validate that all embeddings have consistent dimensions."""
        if not embeddings:
            return

        # Check all embeddings have same dimension
        dims = [len(emb) for emb in embeddings]
        if len(set(dims)) > 1:
            raise EmbeddingError(
                f"Inconsistent embedding dimensions: {set(dims)}. "
                f"All embeddings must have the same dimension."
            )

        # Validate against expected dimension
        actual_dim = dims[0]
        if self.embedding_dim and actual_dim != self.embedding_dim:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {actual_dim}"
            )

        # Update embedding dimension if not set
        if not self.embedding_dim:
            self.embedding_dim = actual_dim

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        if self.embedding_dim is None:
            raise EmbeddingError("Embedding dimension not yet determined. Run embed() first.")
        return self.embedding_dim
