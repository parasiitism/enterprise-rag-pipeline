from enterprise_rag.embeddings import (
    EmbeddingProviderInfo,
    EmbeddingRequest,
    EmbeddingResult,
)


def test_embedding_request_stores_record_text_and_metadata() -> None:
    request = EmbeddingRequest(
        record_id="chunk_1",
        text="Python is a programming language.",
        metadata={"document_id": "doc_1", "title": "Python"},
    )

    assert request.record_id == "chunk_1"
    assert request.text == "Python is a programming language."
    assert request.metadata["document_id"] == "doc_1"


def test_embedding_result_stores_vector_lineage() -> None:
    result = EmbeddingResult(
        record_id="chunk_1",
        vector=[0.1, 0.2, 0.3],
        provider="test",
        model="test-embedding-model",
        dimensions=3,
        metadata={"document_id": "doc_1"},
    )

    assert result.record_id == "chunk_1"
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.provider == "test"
    assert result.model == "test-embedding-model"
    assert result.dimensions == 3


def test_embedding_provider_info_describes_provider_limits() -> None:
    info = EmbeddingProviderInfo(
        provider="gemini",
        model="text-embedding-004",
        dimensions=768,
        max_batch_size=100,
    )

    assert info.provider == "gemini"
    assert info.model == "text-embedding-004"
    assert info.dimensions == 768
    assert info.max_batch_size == 100
