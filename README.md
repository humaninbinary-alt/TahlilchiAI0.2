# TahlilchiAI - Uzbek-Aware RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with comprehensive support for Uzbek language, multi-domain knowledge, and enterprise features.

## Features

### 🌍 Uzbek Language Support
- **UzbekNormalizer**: Handles apostrophe variants (o', g'), digraphs, and Uzbek-specific text normalization
- **Light Stemming**: 16+ common Uzbek suffixes
- **Domain Synonyms**: HR, Legal, Finance/Banking terminology in Uzbek

### 🔍 Advanced Retrieval
- **Hybrid Search**: Dense vector search + lexical overlap boosting
- **Query Expansion**: Domain-specific synonym expansion
- **Reciprocal Rank Fusion (RRF)**: Multi-query result fusion with k=60
- **Semantic Reranking**: Cosine similarity-based reranking

### 📚 Document Processing
- **Multi-format Support**: PDF, DOCX, TXT
- **Smart Chunking**: Heading detection (Uzbek/Russian/English), bullet lists, paragraph boundaries
- **Metadata Preservation**: Document title, section, language, page numbers

### 🎯 Enterprise-Ready
- **Multi-tenancy**: Organization-based filtering
- **AssistantConfig**: Configurable behavior (domain, tone, style, sensitivity)
- **Qdrant Integration**: Scalable vector database with filtering
- **LLM Abstraction**: Pluggable LLM clients (OpenAI, Claude, Gemini)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query → Normalize → Expand → Retrieve → Rerank → Prompt → LLM │
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐        │
│  │ UzbekNorm    │→ │ QueryExpander │→ │ Hybrid       │        │
│  │              │  │ (synonyms)    │  │ Retriever    │        │
│  └──────────────┘  └───────────────┘  │ - Vector     │        │
│                                        │ - RRF        │        │
│                                        │ - Lexical    │        │
│                                        └──────────────┘        │
│                           ↓                                     │
│                    ┌──────────────┐                            │
│                    │ Reranker     │                            │
│                    │ (cosine sim) │                            │
│                    └──────────────┘                            │
│                           ↓                                     │
│                    ┌──────────────┐                            │
│                    │ PromptBuilder│                            │
│                    │ (config-aware)│                            │
│                    └──────────────┘                            │
│                           ↓                                     │
│                    ┌──────────────┐                            │
│                    │ LLMClient    │                            │
│                    │ (OpenAI/etc) │                            │
│                    └──────────────┘                            │
│                           ↓                                     │
│                  {answer, sources, metadata}                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd TahlilchiAI0.2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Qdrant

#### Option A: Docker (Recommended)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option B: Cloud
Sign up at [Qdrant Cloud](https://cloud.qdrant.io/) and get your API key.

### 4. Configure environment variables

```bash
# Required for LLM
export OPENAI_API_KEY=your_openai_api_key

# Optional: Qdrant configuration
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your_qdrant_api_key  # if using cloud
```

## Quick Start

### Run the Playground Demo

```bash
python playground.py
```

This will:
1. Ingest the demo HR policy document into Qdrant
2. Run a sample query: "Ta'til necha kun bo'lishi mumkin?"
3. Display the answer and sources

### Expected Output

```
======================================================================
RAG SYSTEM PLAYGROUND
======================================================================

>>> STEP 1: INGESTING DOCUMENT INTO QDRANT
  → Extracting text from ./data/demo_hr_policy_uzbek.txt...
  → Created 15 chunks
  ✓ Ingestion complete

>>> STEP 2: SETTING UP RAG COMPONENTS
  ✓ Components initialized

>>> STEP 3: RUNNING RAG PIPELINE
  Query: Ta'til necha kun bo'lishi mumkin?

======================================================================
ANSWER
======================================================================
Barcha xodimlar yiliga kamida 24 ish kunidan iborat asosiy mehnat
ta'tilini olish huquqiga ega...

======================================================================
SOURCES (5 chunks used)
======================================================================
[1] Document: Demo HR Policy (Uzbek)
    Section: 1. TA'TIL MUDDATI VA TURLARI
    ...
```

## Usage

### Basic RAG Query

```python
from rag.pipeline.rag_answer import rag_answer
from rag.llm.openai_client import OpenAILLMClient
from rag.models.assistant_config import AssistantConfig, RetrievalPrefs

# Configure assistant
config = AssistantConfig(
    id="assistant_1",
    org_id="org_1",
    name="HR Assistant",
    domain="HR",
    use_case="Answer HR questions",
    sensitivity="MEDIUM",
    tone="professional",
    style="concise",
    primary_language="uz",
    retrieval_prefs=RetrievalPrefs(
        use_hybrid=True,
        use_reranker=True,
        max_context_chunks=5
    )
)

# Initialize components (see playground.py for full setup)
# retriever, reranker, prompt_builder, llm_client = ...

# Run query
result = rag_answer(
    query="Ta'til necha kun bo'lishi mumkin?",
    config=config,
    retriever=retriever,
    llm_client=llm_client,
    prompt_builder=prompt_builder,
    reranker=reranker
)

print(result["answer"])
print(result["sources"])
```

### Document Ingestion

```python
from rag.ingestion.text_extractor import TextExtractor
from rag.ingestion.chunker import Chunker
from rag.ingestion.normalizer import UzbekNormalizer
from rag.ingestion.embedder import Embedder
from rag.ingestion.upsert import QdrantUpserter, ingest_document
from rag.models.chunk import Chunk

# Setup
extractor = TextExtractor()
normalizer = UzbekNormalizer()
chunker = Chunker(normalizer)
embedder = Embedder(model_name="all-MiniLM-L6-v2", backend="local")
upserter = QdrantUpserter(
    collection_name="my_collection",
    url="http://localhost:6333"
)

# Extract and chunk
text = extractor.extract("path/to/document.pdf")
raw_chunks = chunker.chunk_text(text, "Document Title", "uz")

# Create Chunk objects
chunks = [
    Chunk(
        id=f"doc_1_chunk_{i}",
        document_id="doc_1",
        org_id="org_1",
        assistant_ids=["assistant_1"],
        text=ch["text"],
        metadata=ch["metadata"]
    )
    for i, ch in enumerate(raw_chunks)
]

# Embed and upsert
ingest_document(chunks, embedder, upserter)
```

## Project Structure

```
TahlilchiAI0.2/
├── rag/
│   ├── models/              # Data classes
│   │   ├── assistant_config.py
│   │   ├── document.py
│   │   └── chunk.py
│   ├── ingestion/           # Document processing
│   │   ├── text_extractor.py
│   │   ├── chunker.py
│   │   ├── normalizer.py
│   │   ├── embedder.py
│   │   └── upsert.py
│   ├── retrieval/           # Search and ranking
│   │   ├── filters.py
│   │   ├── query_expander.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── prompts/             # Prompt engineering
│   │   └── builder.py
│   ├── pipeline/            # Orchestration
│   │   └── rag_answer.py
│   └── llm/                 # LLM clients
│       └── openai_client.py
├── data/                    # Sample documents
│   └── demo_hr_policy_uzbek.txt
├── playground.py            # Demo script
├── requirements.txt
└── README.md
```

## Configuration

### AssistantConfig Parameters

- **domain**: `"HR"`, `"Legal"`, `"Finance"`, `"Banking"`
- **tone**: `"professional"`, `"friendly"`, `"formal"`
- **style**: `"concise"`, `"step_by_step"`, `"detailed"`
- **sensitivity**: `"LOW"`, `"MEDIUM"`, `"HIGH"`
- **primary_language**: `"uz"`, `"ru"`, `"en"`

### Retrieval Parameters

- **use_hybrid**: Enable hybrid search (vector + lexical)
- **use_reranker**: Enable semantic reranking
- **max_context_chunks**: Number of chunks to include in prompt (default: 5)
- **max_tokens_answer**: Maximum LLM response tokens (default: 800)

## Supported File Formats

- **PDF**: via `pdfminer.six`
- **DOCX**: via `python-docx`
- **TXT**: with multi-encoding support (UTF-8, CP1251, Latin-1)

## Performance

- **Embedding**: ~32 texts/batch (configurable)
- **Qdrant Upsert**: 200 points/batch with retry logic
- **Retrieval**: Multi-query with RRF fusion
- **Reranking**: Batch cosine similarity computation

## Supported Languages

- **Primary**: Uzbek (uz)
- **Secondary**: Russian (ru), English (en)
- **Extensible**: Add new languages via normalizer and synonyms

## Dependencies

### Core
- `pdfminer.six` - PDF text extraction
- `python-docx` - DOCX processing
- `sentence-transformers` - Local embeddings
- `qdrant-client` - Vector database
- `openai` - LLM API

### Optional
- `cohere` - Cohere embeddings
- `anthropic` - Claude API

## License

[Your License Here]

## Contributing

[Contribution guidelines]

## Support

For issues and questions, please open an issue on GitHub.

---

Built with ❤️ for Uzbek language processing
