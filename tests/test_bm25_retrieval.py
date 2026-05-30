import pytest

from enterprise_rag.retrieval.bm25 import BM25ChunkIndex


def test_bm25_chunk_index_returns_matching_chunk() -> None:
    records = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "source": "test",
            "source_id": "1",
            "title": "Python",
            "chunk_index": 0,
            "text": "Python is a programming language.",
            "metadata": {"page_id": "1"},
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_2",
            "source": "test",
            "source_id": "2",
            "title": "Cooking",
            "chunk_index": 0,
            "text": "Cooking pasta requires boiling water.",
            "metadata": {"page_id": "2"},
        },
    ]

    index = BM25ChunkIndex(records)
    results = index.search("python programming", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"
    assert results[0].document_id == "doc_1"
    assert results[0].title == "Python"


def test_bm25_chunk_index_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="chunk_records cannot be empty"):
        BM25ChunkIndex([])


def test_bm25_search_rejects_invalid_top_k() -> None:
    records = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "source": "test",
            "source_id": "1",
            "title": "Python",
            "chunk_index": 0,
            "text": "Python is a programming language.",
            "metadata": {},
        }
    ]

    index = BM25ChunkIndex(records)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        index.search("python", top_k=0)