from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    title: str
    score: float
    text: str
    metadata: dict[str, object]
