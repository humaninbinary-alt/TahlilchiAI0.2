"""
RAG System Playground Script

This script demonstrates the complete RAG pipeline:
1. Ingests a demo document into Qdrant
2. Runs a query through the full pipeline
3. Displays answer and sources

Usage:
    export OPENAI_API_KEY=your_key_here
    python playground.py
"""

import os
from typing import List

from qdrant_client import QdrantClient

from rag.models.assistant_config import AssistantConfig, RetrievalPrefs
from rag.models.document import Document
from rag.models.chunk import Chunk
from rag.ingestion.text_extractor import TextExtractor
from rag.ingestion.chunker import Chunker
from rag.ingestion.normalizer import UzbekNormalizer
from rag.ingestion.embedder import Embedder
from rag.ingestion.upsert import QdrantUpserter, ingest_document
from rag.retrieval.query_expander import QueryExpander
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.prompts.builder import PromptBuilder
from rag.pipeline.rag_answer import rag_answer
from rag.llm.openai_client import OpenAILLMClient


def build_dummy_assistant() -> AssistantConfig:
    """Build a demo HR assistant configuration."""
    return AssistantConfig(
        id="assistant_hr_demo",
        org_id="org_demo",
        name="HR Demo Assistant",
        domain="HR",
        use_case="Answer HR policy questions",
        sensitivity="MEDIUM",
        tone="professional",
        style="step_by_step",
        primary_language="uz",
        secondary_languages=["ru", "en"],
        allowed_tags=["HR", "Policies"],
        attached_doc_ids=[],
        retrieval_prefs=RetrievalPrefs(
            use_hybrid=True,
            use_reranker=True,
            max_context_chunks=5,
            max_tokens_answer=800,
        ),
    )


def ingest_single_document(
    doc_path: str,
    doc_title: str,
    assistant: AssistantConfig,
    collection_name: str,
    qdrant_url: str,
    qdrant_api_key: str,
):
    """
    Extract, chunk, embed, and upsert a single document into Qdrant.

    Args:
        doc_path: Path to the document file
        doc_title: Title of the document
        assistant: AssistantConfig for metadata
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key (optional)
    """
    # Setup helpers
    extractor = TextExtractor()
    normalizer = UzbekNormalizer()
    chunker = Chunker(normalizer=normalizer)
    embedder = Embedder(model_name="all-MiniLM-L6-v2", backend="local")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    upserter = QdrantUpserter(
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Build document object
    document = Document(
        id="doc_demo",
        org_id=assistant.org_id,
        title=doc_title,
        path=doc_path,
        tags=["HR", "Policies"],
        language=assistant.primary_language,
        is_global=True,
    )

    # 1) Extract text
    print(f"  → Extracting text from {doc_path}...")
    raw_text = extractor.extract(document.path)
    print(f"  → Extracted {len(raw_text)} characters")

    # 2) Chunk
    print(f"  → Chunking text...")
    raw_chunks = chunker.chunk_text(
        text=raw_text,
        document_title=document.title,
        language=document.language,
    )
    print(f"  → Created {len(raw_chunks)} chunks")

    # 3) Wrap into Chunk dataclasses
    chunks: List[Chunk] = []
    for i, ch in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                org_id=document.org_id,
                assistant_ids=[assistant.id],
                text=ch["text"],
                metadata=ch["metadata"],
                embedding_vector=[],
                sparse_tokens=[],
            )
        )

    # 4) Embed + upsert into Qdrant
    print(f"  → Embedding and upserting to Qdrant...")
    ingest_document(chunks=chunks, embedder=embedder, upserter=upserter)


def main():
    """Main playground function."""
    print("=" * 70)
    print("RAG SYSTEM PLAYGROUND")
    print("=" * 70)

    # === CONFIG ===
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", None)
    collection_name = "rag_demo_collection"

    # Path to a local Uzbek HR/Policy document you want to test with
    # e.g., "./data/hr_policy_uzbek.pdf"
    doc_path = "./data/demo_hr_policy_uzbek.txt"
    doc_title = "Demo HR Policy (Uzbek)"

    # Check if document exists
    if not os.path.exists(doc_path):
        print(f"\n[ERROR] Demo document not found at: {doc_path}")
        print("Please create a simple Uzbek text file at that path and re-run.")
        print("\nExample content:")
        print("-" * 70)
        print("""
# MEHNAT TA'TILI QOIDALARI

## 1. TA'TIL MUDDATI
Barcha xodimlar yiliga 24 ish kunidan kam bo'lmagan ta'til olish huquqiga ega.

## 2. TA'TIL REJALASHTIRISH
Ta'til grafigi har yil yanvar oyida tuziladi. Xodimlar o'z ta'til kunlarini
oldindan rejalashtirishi kerak.

## 3. TA'TIL TO'LOVI
Ta'til davomida xodimlar o'rtacha oylik maoshiga teng to'lov oladilar.
        """)
        print("-" * 70)
        return

    # Build assistant config
    assistant = build_dummy_assistant()

    # === INGESTION ===
    print(f"\n>>> STEP 1: INGESTING DOCUMENT INTO QDRANT")
    print(f"  Collection: {collection_name}")
    print(f"  Qdrant URL: {qdrant_url}")

    try:
        ingest_single_document(
            doc_path=doc_path,
            doc_title=doc_title,
            assistant=assistant,
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key or "",
        )
        print("  ✓ Ingestion complete")
    except Exception as e:
        print(f"  ✗ Ingestion failed: {str(e)}")
        return

    # === SETUP RAG RUNTIME COMPONENTS ===
    print(f"\n>>> STEP 2: SETTING UP RAG COMPONENTS")

    embedder = Embedder(model_name="all-MiniLM-L6-v2", backend="local")
    query_expander = QueryExpander()
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    retriever = HybridRetriever(
        client=qdrant_client,
        collection_name=collection_name,
        embedder=embedder,
        query_expander=query_expander,
    )
    reranker = Reranker(embedder=embedder)
    prompt_builder = PromptBuilder()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not set in environment")
        print("Please set it with: export OPENAI_API_KEY=your_key_here")
        return

    llm_client = OpenAILLMClient(model="gpt-4o-mini")
    print("  ✓ Components initialized")

    # === TEST QUERY ===
    query = "Ta'til necha kun bo'lishi mumkin?"

    print(f"\n>>> STEP 3: RUNNING RAG PIPELINE")
    print(f"  Query: {query}")
    print(f"  Language: {assistant.primary_language}")
    print(f"  Domain: {assistant.domain}")

    try:
        result = rag_answer(
            query=query,
            config=assistant,
            retriever=retriever,
            llm_client=llm_client,
            prompt_builder=prompt_builder,
            reranker=reranker,
            history=None,
            top_k=10,
        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "=" * 70)
        print(f"SOURCES ({result['used_chunk_count']} chunks used)")
        print("=" * 70)
        for i, src in enumerate(result["sources"], 1):
            print(f"\n[{i}] Document: {src['document_title']}")
            print(f"    Section: {src.get('section_title', 'N/A')}")
            print(f"    Chunk ID: {src['chunk_id']}")
            print(f"    Preview: {src['text_preview'][:120]!r}...")

        print("\n" + "=" * 70)
        print("✓ RAG PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ RAG pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
