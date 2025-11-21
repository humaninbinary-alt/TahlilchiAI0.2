from typing import List
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.ingestion.embedder import Embedder
from rag.models.chunk import Chunk
from rag.ingestion.normalizer import normalize_uzbek
from rag.models.assistant_config import AssistantConfig


class Reranker:
    """
    Semantic reranker using cosine similarity.

    Features:
    - Reranks chunks based on semantic similarity to query
    - Uses embedder for query and chunk vectors
    - Computes cosine similarity scores
    - Returns top_k most relevant chunks
    """

    def __init__(self, embedder: Embedder):
        """
        Initialize reranker with embedder.

        Args:
            embedder: Embedder instance for generating vectors
        """
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        config: AssistantConfig,
        chunks: List[Chunk],
        top_k: int,
    ) -> List[Chunk]:
        """
        Rerank chunks by semantic similarity to query.

        Process:
        1. Normalize query
        2. Embed query
        3. Embed all chunk texts
        4. Compute cosine similarity for each chunk
        5. Sort by similarity (descending)
        6. Return top_k chunks

        Args:
            query: User query string
            config: AssistantConfig (currently unused but available for future features)
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return

        Returns:
            List of top_k chunks sorted by relevance
        """
        # Handle empty input
        if not chunks:
            return []

        # Handle top_k larger than chunks
        if top_k >= len(chunks):
            return chunks

        try:
            # Step 1: Normalize query
            normalized_query = normalize_uzbek(query)

            # Step 2: Embed query
            query_embedding = self.embedder.embed([normalized_query])[0]

            # Step 3: Embed all chunk texts
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = self.embedder.embed(chunk_texts)

            # Step 4: Compute cosine similarity for each chunk
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                cos_sim = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((cos_sim, i, chunks[i]))

            # Step 5: Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Step 6: Return top_k chunks
            reranked_chunks = [chunk for _, _, chunk in similarities[:top_k]]

            return reranked_chunks

        except Exception as e:
            # If reranking fails, return original chunks unchanged
            print(f"Warning: Reranking failed ({str(e)}), returning original chunks")
            return chunks[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Formula: cos_sim = dot(v1, v2) / (||v1|| * ||v2||)

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (between -1 and 1, typically 0 to 1 for embeddings)
        """
        # Ensure vectors have same dimension
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Compute magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Compute cosine similarity
        cos_sim = dot_product / (mag1 * mag2)

        return cos_sim
