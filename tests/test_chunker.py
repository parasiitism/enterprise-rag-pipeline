import pytest
from langchain_core.documents import Document

from enterprise_rag.chunker import chunk_canonical_document
from enterprise_rag.documents import CanonicalDocument


def test_chunk_canonical_document_creates_traceable_chunks() -> None:
    document = CanonicalDocument(
        document_id="wikipedia_1",
        source="wikipedia",
        source_id="1",
        title="Test Page",
        version="100",
        updated_at="2026-01-01T00:00:00Z",
        text=" ".join(f"word{i}" for i in range(500)),
        metadata={
            "page_id": "1",
            "revision_id": "100",
        },
    )

    chunks = chunk_canonical_document(
        document=document,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)

    first_chunk = chunks[0]

    assert first_chunk.metadata["chunk_id"] == "wikipedia_1_chunk_0"
    assert first_chunk.metadata["chunk_index"] == 0
    assert first_chunk.metadata["document_id"] == "wikipedia_1"
    assert first_chunk.metadata["source"] == "wikipedia"
    assert first_chunk.metadata["source_id"] == "1"
    assert first_chunk.metadata["title"] == "Test Page"
    assert first_chunk.metadata["version"] == "100"
    assert first_chunk.metadata["page_id"] == "1"
    assert first_chunk.metadata["revision_id"] == "100"


def test_chunk_canonical_document_rejects_invalid_overlap() -> None:
    document = CanonicalDocument(
        document_id="wikipedia_1",
        source="wikipedia",
        source_id="1",
        title="Test Page",
        version="100",
        updated_at="2026-01-01T00:00:00Z",
        text="hello world",
        metadata={},
    )

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        chunk_canonical_document(
            document=document,
            chunk_size=100,
            chunk_overlap=100,
        )