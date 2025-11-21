from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Chunk:
    id: str
    document_id: str
    org_id: str
    assistant_ids: List[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: List[float] = field(default_factory=list)
    sparse_tokens: List[str] = field(default_factory=list)
