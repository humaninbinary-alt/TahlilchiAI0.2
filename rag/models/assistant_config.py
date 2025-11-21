from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RetrievalPrefs:
    use_hybrid: bool = True
    use_reranker: bool = True
    max_context_chunks: int = 5
    max_tokens_answer: int = 800

@dataclass
class AssistantConfig:
    id: str
    org_id: str
    name: str
    domain: str
    use_case: str
    sensitivity: str
    tone: str
    style: str
    primary_language: str = "uz"
    secondary_languages: List[str] = field(default_factory=lambda: ["ru", "en"])
    allowed_tags: List[str] = field(default_factory=list)
    attached_doc_ids: List[str] = field(default_factory=list)
    retrieval_prefs: RetrievalPrefs = field(default_factory=RetrievalPrefs)
