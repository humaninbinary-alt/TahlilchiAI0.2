from typing import List, Dict, Tuple, Any
from collections import defaultdict

from qdrant_client import QdrantClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.assistant_config import AssistantConfig
from rag.models.chunk import Chunk
from rag.retrieval.filters import build_qdrant_filter
from rag.retrieval.query_expander import QueryExpander
from rag.ingestion.normalizer import normalize_uzbek
from rag.ingestion.embedder import Embedder


class HybridRetriever:
    """
    Production-ready hybrid retrieval system with RRF and lexical boosting.

    Features:
    - Query expansion with domain-specific synonyms
    - Dense vector search using embeddings
    - Reciprocal Rank Fusion (RRF) for combining results
    - Lexical overlap boosting for hybrid scoring
    - Qdrant filtering by org, language, and assistant
    """

    # RRF constant for rank fusion
    RRF_K = 60

    # Weight for lexical overlap boost
    LEXICAL_BOOST_WEIGHT = 0.2

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: Embedder,
        query_expander: QueryExpander,
    ):
        """
        Initialize hybrid retriever.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to search
            embedder: Embedder for generating query vectors
            query_expander: QueryExpander for query variants
        """
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.query_expander = query_expander

    def retrieve(
        self,
        query: str,
        config: AssistantConfig,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks using hybrid search with RRF.

        Pipeline:
        1. Normalize and expand query into variants
        2. Embed each query variant
        3. Search Qdrant for each variant
        4. Apply Reciprocal Rank Fusion (RRF)
        5. Add lexical overlap boost
        6. Sort by final score and return top_k

        Args:
            query: User query string
            config: AssistantConfig for filtering and expansion
            top_k: Number of chunks to return
            score_threshold: Minimum similarity score (optional)

        Returns:
            List of Chunk objects, sorted by relevance
        """
        if not query or not query.strip():
            return []

        # Step 1: Normalize original query for lexical matching
        normalized_query = normalize_uzbek(query)

        # Step 2: Expand query into variants
        query_variants = self.query_expander.expand(query, config)

        # Step 3: Build Qdrant filter
        query_filter = build_qdrant_filter(config)

        # Step 4: Search for each query variant
        all_results = []
        for variant_idx, query_variant in enumerate(query_variants):
            try:
                # Embed the query variant
                query_vector = self.embedder.embed([query_variant])[0]

                # Search Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k * 2,  # Retrieve more for better fusion
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )

                # Store results with variant index
                for rank, result in enumerate(search_results):
                    all_results.append({
                        'point': result,
                        'variant_idx': variant_idx,
                        'rank': rank,
                        'score': result.score,
                    })

            except Exception as e:
                # Log error but continue with other variants
                print(f"Warning: Search failed for variant '{query_variant}': {str(e)}")
                continue

        if not all_results:
            return []

        # Step 5: Apply Reciprocal Rank Fusion (RRF)
        fused_results = self._apply_rrf(all_results)

        # Step 6: Add lexical overlap boost
        boosted_results = self._add_lexical_boost(
            fused_results,
            normalized_query
        )

        # Step 7: Sort by final score (descending)
        sorted_results = sorted(
            boosted_results.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )

        # Step 8: Build Chunk objects and return top_k
        chunks = []
        for point_id, result_data in sorted_results[:top_k]:
            try:
                chunk = self._build_chunk_from_payload(
                    point_id=point_id,
                    payload=result_data['payload']
                )
                chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Failed to build chunk from point {point_id}: {str(e)}")
                continue

        return chunks

    def _apply_rrf(self, all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple queries.

        RRF formula: score = Σ (1 / (k + rank)) for each query variant

        Args:
            all_results: List of search results from all query variants

        Returns:
            Dictionary of {point_id: {payload, rrf_score, ranks}}
        """
        fused = defaultdict(lambda: {
            'payload': None,
            'rrf_score': 0.0,
            'ranks': [],
        })

        for result in all_results:
            point = result['point']
            point_id = str(point.id)
            rank = result['rank']
            variant_idx = result['variant_idx']

            # Update payload (use first occurrence)
            if fused[point_id]['payload'] is None:
                fused[point_id]['payload'] = point.payload

            # Add to RRF score
            rrf_contribution = 1.0 / (self.RRF_K + rank)
            fused[point_id]['rrf_score'] += rrf_contribution

            # Track ranks for debugging
            fused[point_id]['ranks'].append((variant_idx, rank))

        return dict(fused)

    def _add_lexical_boost(
        self,
        fused_results: Dict[str, Dict[str, Any]],
        normalized_query: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Add lexical overlap boost to RRF scores.

        Overlap score = |intersection(query_tokens, text_tokens)| / (1 + |query_tokens|)
        Final score = rrf_score + (LEXICAL_BOOST_WEIGHT * overlap_score)

        Args:
            fused_results: Results after RRF
            normalized_query: Normalized query string

        Returns:
            Updated results with final_score
        """
        # Tokenize normalized query
        query_tokens = set(normalized_query.split())

        if not query_tokens:
            # No tokens, just use RRF score
            for point_id in fused_results:
                fused_results[point_id]['final_score'] = fused_results[point_id]['rrf_score']
            return fused_results

        # Calculate overlap for each result
        for point_id, result_data in fused_results.items():
            payload = result_data['payload']

            # Get text from payload
            text = payload.get('text', '')
            if not text:
                overlap_score = 0.0
            else:
                # Normalize and tokenize the text
                normalized_text = normalize_uzbek(text)
                text_tokens = set(normalized_text.split())

                # Calculate overlap
                intersection = query_tokens.intersection(text_tokens)
                overlap_score = len(intersection) / (1 + len(query_tokens))

            # Compute final score
            rrf_score = result_data['rrf_score']
            final_score = rrf_score + (self.LEXICAL_BOOST_WEIGHT * overlap_score)

            result_data['overlap_score'] = overlap_score
            result_data['final_score'] = final_score

        return fused_results

    def _build_chunk_from_payload(
        self,
        point_id: str,
        payload: Dict[str, Any]
    ) -> Chunk:
        """
        Build Chunk object from Qdrant payload.

        Args:
            point_id: Qdrant point ID
            payload: Point payload containing chunk data

        Returns:
            Chunk object
        """
        # Extract core fields
        document_id = payload.get('document_id', '')
        org_id = payload.get('org_id', '')
        assistant_ids = payload.get('assistant_ids', [])
        text = payload.get('text', '')
        sparse_tokens = payload.get('sparse_tokens', [])

        # Build metadata from remaining fields
        metadata_keys = {
            'document_id', 'org_id', 'assistant_ids', 'text', 'sparse_tokens'
        }
        metadata = {
            k: v for k, v in payload.items()
            if k not in metadata_keys
        }

        # Create Chunk object
        chunk = Chunk(
            id=point_id,
            document_id=document_id,
            org_id=org_id,
            assistant_ids=assistant_ids,
            text=text,
            metadata=metadata,
            embedding_vector=[],  # Not needed for retrieval results
            sparse_tokens=sparse_tokens,
        )

        return chunk
