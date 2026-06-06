from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.vector_store.models import (
    VectorIndexInfo,
    VectorRecord,
    VectorSearchResult,
)


class VectorIndex(Protocol):
    @property
    def info(self) -> VectorIndexInfo:
        raise NotImplementedError

    def add(self, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError