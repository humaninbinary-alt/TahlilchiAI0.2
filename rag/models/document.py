from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    id: str
    org_id: str
    title: str
    path: str
    tags: List[str]
    language: str = "uz"
    is_global: bool = False
