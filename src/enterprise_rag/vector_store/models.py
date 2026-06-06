from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


VectorMetric = Literal["cosine", "l2", "ip"]


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    chunk_id: str
    document_id: str
    title: str
    vector: list[float]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    record: VectorRecord
    score: float
    rank: int


@dataclass(frozen=True)
class VectorIndexInfo:
    name: str
    dimensions: int
    metric: VectorMetric
    records_count: int