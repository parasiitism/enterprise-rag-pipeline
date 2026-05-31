from __future__ import annotations # modern hints safely

from dataclasses import dataclass

from enterprise_rag.retrieval.models import SearchResult

# This represent one evaluation question
@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    expected_document_id: str
    expected_chunk_id: str | None = None

# This represents one evaluation result
@dataclass(frozen=True)
class RetrievalEvalResult:
    case: RetrievalEvalCase
    retrieved_results: list[SearchResult]
    top_k: int
    hit: bool
    matched_document_id: str | None
    matched_chunk_id: str | None

# This stores the final score across all eval cases
@dataclass(frozen=True)
class RetrievalEvalSummary:
    total_cases: int
    hits: int
    misses: int
    hit_rate: float