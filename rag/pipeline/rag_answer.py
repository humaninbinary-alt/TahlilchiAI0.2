from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.assistant_config import AssistantConfig
from rag.models.chunk import Chunk
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.prompts.builder import PromptBuilder


class LLMClient(ABC):
    """
    Abstract interface for LLM providers.

    Implementations can use OpenAI, Claude, Gemini, or any other LLM service.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a full prompt string.

        Args:
            prompt: Complete prompt string with system instructions, context, and query

        Returns:
            Generated answer string from the LLM

        Raises:
            Exception: If LLM generation fails
        """
        raise NotImplementedError("LLMClient.generate() must be implemented by subclass")


def rag_answer(
    query: str,
    config: AssistantConfig,
    retriever: HybridRetriever,
    llm_client: LLMClient,
    prompt_builder: PromptBuilder,
    reranker: Optional[Reranker] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Complete RAG pipeline orchestration.

    Pipeline:
    1. Retrieve candidate chunks from vector database
    2. Optionally rerank chunks by semantic similarity
    3. Build context-aware prompt with AssistantConfig
    4. Generate answer using LLM
    5. Prepare sources and metadata

    Args:
        query: User's question
        config: AssistantConfig for filtering, prompt building, and behavior
        retriever: HybridRetriever for vector search
        llm_client: LLMClient implementation for answer generation
        prompt_builder: PromptBuilder for creating prompts
        reranker: Optional Reranker for semantic reranking
        history: Optional chat history as [{"role": "user"/"assistant", "content": "..."}]
        top_k: Number of chunks to retrieve initially

    Returns:
        Dictionary containing:
        - answer: Generated answer string
        - sources: List of source chunk metadata
        - used_chunk_count: Number of chunks used in context

    Raises:
        Exception: If any component fails critically

    Example:
        result = rag_answer(
            query="Ta'til necha kun bo'lishi mumkin?",
            config=assistant_config,
            retriever=hybrid_retriever,
            llm_client=my_llm_client,
            prompt_builder=PromptBuilder(),
        )
        print(result["answer"])
        print(result["sources"])
    """
    try:
        # Step 1: Retrieve candidate chunks
        candidate_chunks = retriever.retrieve(
            query=query,
            config=config,
            top_k=top_k
        )

        # Step 2: Rerank if enabled and reranker provided
        if config.retrieval_prefs.use_reranker and reranker is not None:
            chunks = reranker.rerank(
                query=query,
                config=config,
                chunks=candidate_chunks,
                top_k=config.retrieval_prefs.max_context_chunks,
            )
        else:
            # Use top chunks without reranking
            chunks = candidate_chunks[:config.retrieval_prefs.max_context_chunks]

        # Step 3: Build prompt
        prompt = prompt_builder.build_prompt(
            query=query,
            config=config,
            chunks=chunks,
            history=history,
        )

        # Step 4: Generate answer using LLM
        try:
            answer = llm_client.generate(prompt)
        except Exception as e:
            # If LLM fails, return error message
            raise Exception(f"LLM generation failed: {str(e)}")

        # Step 5: Prepare sources list
        sources = _build_sources_list(chunks)

        # Step 6: Return result
        return {
            "answer": answer,
            "sources": sources,
            "used_chunk_count": len(chunks),
        }

    except Exception as e:
        # Log error and re-raise or return error result
        print(f"Error in rag_answer: {str(e)}")
        raise


def _build_sources_list(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """
    Build list of source metadata from chunks.

    Args:
        chunks: List of chunks used in context

    Returns:
        List of source dictionaries with chunk metadata
    """
    sources = []

    for chunk in chunks:
        # Build source entry
        source = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "org_id": chunk.org_id,
            "text_preview": chunk.text[:200] if chunk.text else "",  # First 200 chars
        }

        # Add optional metadata fields
        source["document_title"] = chunk.metadata.get("document_title")
        source["section_title"] = chunk.metadata.get("section_title")
        source["page_number"] = chunk.metadata.get("page_number")
        source["language"] = chunk.metadata.get("language")
        source["chunk_index"] = chunk.metadata.get("chunk_index")

        sources.append(source)

    return sources
