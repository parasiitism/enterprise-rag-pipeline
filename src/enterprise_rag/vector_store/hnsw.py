from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from enterprise_rag.vector_store.models import (
    VectorIndexInfo,
    VectorMetric,
    VectorRecord,
    VectorSearchResult,
)


DEFAULT_HNSW_MAX_ELEMENTS = 10_000
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_SEARCH = 50
SUPPORTED_HNSW_METRICS: set[VectorMetric] = {"cosine", "l2", "ip"}


class HnswIndexLike(Protocol):
    def init_index(
        self,
        *,
        max_elements: int,
        ef_construction: int,
        M: int,
    ) -> None:
        raise NotImplementedError

    def add_items(self, data: Any, ids: Any) -> None:
        raise NotImplementedError

    def knn_query(self, data: Any, k: int) -> tuple[Any, Any]:
        raise NotImplementedError

    def set_ef(self, ef: int) -> None:
        raise NotImplementedError

    def resize_index(self, new_size: int) -> None:
        raise NotImplementedError


class HnswIndexFactory(Protocol):
    def __call__(self, *, space: str, dim: int) -> HnswIndexLike:
        raise NotImplementedError


class HNSWVectorIndex:
    def __init__(
        self,
        dimensions: int,
        metric: VectorMetric = "cosine",
        max_elements: int = DEFAULT_HNSW_MAX_ELEMENTS,
        ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
        m: int = DEFAULT_HNSW_M,
        ef_search: int = DEFAULT_HNSW_EF_SEARCH,
        index_factory: HnswIndexFactory | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        if max_elements <= 0:
            raise ValueError("max_elements must be greater than 0")

        if ef_construction <= 0:
            raise ValueError("ef_construction must be greater than 0")

        if m <= 0:
            raise ValueError("m must be greater than 0")

        if ef_search <= 0:
            raise ValueError("ef_search must be greater than 0")

        if metric not in SUPPORTED_HNSW_METRICS:
            raise ValueError(f"unsupported HNSW metric: {metric}")

        self._dimensions = dimensions
        self._metric = metric
        self._capacity = max_elements
        self._records: list[VectorRecord] = []
        self._record_id_to_label: dict[str, int] = {}
        self._label_to_record: dict[int, VectorRecord] = {}
        self._next_label = 0

        factory = index_factory or _load_hnsw_index_factory()
        self._index = factory(space=metric, dim=dimensions)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m,
        )
        self._index.set_ef(ef_search)

    @property
    def info(self) -> VectorIndexInfo:
        return VectorIndexInfo(
            name="hnsw",
            dimensions=self._dimensions,
            metric=self._metric,
            records_count=len(self._records),
        )

    def add(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return

        for record in records:
            self._validate_record(record)

            if record.record_id in self._record_id_to_label:
                raise ValueError(f"duplicate vector record_id: {record.record_id}")

        self._ensure_capacity(len(records))

        labels = list(range(self._next_label, self._next_label + len(records)))
        vectors = np.array([record.vector for record in records], dtype=np.float32)

        self._index.add_items(vectors, np.array(labels, dtype=np.int64))

        for label, record in zip(labels, records, strict=True):
            self._records.append(record)
            self._record_id_to_label[record.record_id] = label
            self._label_to_record[label] = record

        self._next_label += len(records)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self._records:
            return []

        if len(query_vector) != self._dimensions:
            raise ValueError(
                f"expected query dimension {self._dimensions}, got {len(query_vector)}"
            )

        effective_top_k = min(top_k, len(self._records))
        query = np.array([query_vector], dtype=np.float32)
        labels, distances = self._index.knn_query(query, k=effective_top_k)

        results: list[VectorSearchResult] = []

        for rank, (label, distance) in enumerate(
            zip(labels[0], distances[0], strict=True),
            start=1,
        ):
            record = self._label_to_record[int(label)]
            score = _distance_to_score(metric=self._metric, distance=float(distance))

            results.append(
                VectorSearchResult(
                    record=record,
                    score=score,
                    rank=rank,
                )
            )

        return results

    def _validate_record(self, record: VectorRecord) -> None:
        if len(record.vector) != self._dimensions:
            raise ValueError(
                f"expected vector dimension {self._dimensions}, "
                f"got {len(record.vector)} for record_id={record.record_id}"
            )

    def _ensure_capacity(self, incoming_count: int) -> None:
        required_capacity = len(self._records) + incoming_count

        if required_capacity <= self._capacity:
            return

        new_capacity = max(required_capacity, self._capacity * 2)
        self._index.resize_index(new_capacity)
        self._capacity = new_capacity


def _distance_to_score(metric: VectorMetric, distance: float) -> float:
    if metric in {"cosine", "ip"}:
        return 1.0 - distance

    return -distance


def _load_hnsw_index_factory() -> HnswIndexFactory:
    try:
        import hnswlib
    except ImportError as exc:
        raise RuntimeError(
            "hnswlib is required for HNSWVectorIndex. "
            'Install it with: python -m pip install -e ".[vector]"'
        ) from exc

    return hnswlib.Index
