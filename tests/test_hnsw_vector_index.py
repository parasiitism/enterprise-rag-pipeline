import pytest

from enterprise_rag.vector_store import HNSWVectorIndex, VectorRecord


class FakeHnswIndex:
    def __init__(self, *, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self.vectors: list[list[float]] = []
        self.labels: list[int] = []
        self.ef: int | None = None
        self.capacity: int | None = None

    def init_index(
        self,
        *,
        max_elements: int,
        ef_construction: int,
        M: int,
    ) -> None:
        self.capacity = max_elements
        self.ef_construction = ef_construction
        self.m = M

    def add_items(self, data, ids) -> None:
        self.vectors.extend(data.tolist())
        self.labels.extend([int(label) for label in ids.tolist()])

    def knn_query(self, data, k: int):
        query = data.tolist()[0]

        scored = [
            (
                label,
                _cosine_distance(query, vector),
            )
            for label, vector in zip(self.labels, self.vectors, strict=True)
        ]
        scored.sort(key=lambda item: item[1])

        labels = [[label for label, _ in scored[:k]]]
        distances = [[distance for _, distance in scored[:k]]]

        return labels, distances

    def set_ef(self, ef: int) -> None:
        self.ef = ef

    def resize_index(self, new_size: int) -> None:
        self.capacity = new_size


def fake_index_factory(*, space: str, dim: int) -> FakeHnswIndex:
    return FakeHnswIndex(space=space, dim=dim)


def _record(record_id: str, vector: list[float]) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        chunk_id=record_id,
        document_id="doc_1",
        title="Test",
        vector=vector,
        metadata={"source": "test"},
    )


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5

    if left_norm == 0 or right_norm == 0:
        return 1.0

    return 1.0 - (dot / (left_norm * right_norm))


def test_hnsw_vector_index_adds_and_searches_records() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        metric="cosine",
        max_elements=10,
        index_factory=fake_index_factory,
    )

    index.add(
        [
            _record("chunk_python", [1.0, 0.0, 0.0]),
            _record("chunk_cooking", [0.0, 1.0, 0.0]),
        ]
    )

    results = index.search([1.0, 0.0, 0.0], top_k=1)

    assert index.info.name == "hnsw"
    assert index.info.dimensions == 3
    assert index.info.metric == "cosine"
    assert index.info.records_count == 2
    assert len(results) == 1
    assert results[0].record.record_id == "chunk_python"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)


def test_hnsw_vector_index_returns_empty_results_when_empty() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        index_factory=fake_index_factory,
    )

    assert index.search([1.0, 0.0, 0.0], top_k=5) == []


def test_hnsw_vector_index_rejects_invalid_top_k() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        index_factory=fake_index_factory,
    )

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        index.search([1.0, 0.0, 0.0], top_k=0)


def test_hnsw_vector_index_rejects_invalid_metric() -> None:
    with pytest.raises(ValueError, match="unsupported HNSW metric"):
        HNSWVectorIndex(
            dimensions=3,
            metric="invalid",  # type: ignore[arg-type]
            index_factory=fake_index_factory,
        )


def test_hnsw_vector_index_rejects_wrong_record_dimension() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        index_factory=fake_index_factory,
    )

    with pytest.raises(ValueError, match="expected vector dimension 3"):
        index.add([_record("chunk_1", [1.0, 0.0])])


def test_hnsw_vector_index_rejects_wrong_query_dimension() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        index_factory=fake_index_factory,
    )
    index.add([_record("chunk_1", [1.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="expected query dimension 3"):
        index.search([1.0, 0.0], top_k=1)


def test_hnsw_vector_index_rejects_duplicate_record_id() -> None:
    index = HNSWVectorIndex(
        dimensions=3,
        index_factory=fake_index_factory,
    )

    index.add([_record("chunk_1", [1.0, 0.0, 0.0])])

    with pytest.raises(ValueError, match="duplicate vector record_id"):
        index.add([_record("chunk_1", [0.0, 1.0, 0.0])])


def test_hnsw_vector_index_resizes_when_capacity_is_exceeded() -> None:
    fake_index = FakeHnswIndex(space="cosine", dim=3)

    def factory(*, space: str, dim: int) -> FakeHnswIndex:
        assert space == "cosine"
        assert dim == 3
        return fake_index

    index = HNSWVectorIndex(
        dimensions=3,
        max_elements=1,
        index_factory=factory,
    )

    index.add(
        [
            _record("chunk_1", [1.0, 0.0, 0.0]),
            _record("chunk_2", [0.0, 1.0, 0.0]),
        ]
    )

    assert fake_index.capacity == 2
    assert index.info.records_count == 2
