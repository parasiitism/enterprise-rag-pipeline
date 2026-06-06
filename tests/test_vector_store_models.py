from collections.abc import Sequence

from enterprise_rag.vector_store import (
    VectorIndexInfo,
    VectorRecord,
    VectorSearchResult,
)


class FakeVectorIndex:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    @property
    def info(self) -> VectorIndexInfo:
        return VectorIndexInfo(
            name="fake",
            dimensions=3,
            metric="cosine",
            records_count=len(self.records),
        )

    def add(self, records: Sequence[VectorRecord]) -> None:
        self.records.extend(records)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                record=record,
                score=1.0,
                rank=index + 1,
            )
            for index, record in enumerate(self.records[:top_k])
        ]


def test_vector_record_stores_embedding_lineage() -> None:
    record = VectorRecord(
        record_id="chunk_1",
        chunk_id="chunk_1",
        document_id="doc_1",
        title="Python",
        vector=[0.1, 0.2, 0.3],
        metadata={"source": "wikipedia"},
    )

    assert record.record_id == "chunk_1"
    assert record.document_id == "doc_1"
    assert record.title == "Python"
    assert record.vector == [0.1, 0.2, 0.3]
    assert record.metadata["source"] == "wikipedia"


def test_vector_index_interface_shape_with_fake_index() -> None:
    index = FakeVectorIndex()
    record = VectorRecord(
        record_id="chunk_1",
        chunk_id="chunk_1",
        document_id="doc_1",
        title="Python",
        vector=[0.1, 0.2, 0.3],
    )

    index.add([record])
    results = index.search([0.1, 0.2, 0.3], top_k=1)

    assert index.info.name == "fake"
    assert index.info.dimensions == 3
    assert index.info.metric == "cosine"
    assert index.info.records_count == 1
    assert len(results) == 1
    assert results[0].record.chunk_id == "chunk_1"
    assert results[0].score == 1.0
    assert results[0].rank == 1