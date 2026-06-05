import pytest

from enterprise_rag.embeddings import (
    DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS,
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    EmbeddingRequest,
    HuggingFaceEmbeddingProvider,
)


class FakeSentenceTransformer:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self.embeddings


class FailingOnceSentenceTransformerLoader:
    def __init__(self, model: FakeSentenceTransformer) -> None:
        self.model = model
        self.calls = 0

    def __call__(self, model_name: str, **model_kwargs: object) -> FakeSentenceTransformer:
        self.calls += 1

        if self.calls == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

        return self.model


def test_huggingface_provider_exposes_default_model_info() -> None:
    provider = HuggingFaceEmbeddingProvider()

    assert provider.info.provider == "huggingface"
    assert provider.info.model == DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
    assert provider.info.dimensions == DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS
    assert provider.info.max_batch_size == 32


def test_huggingface_provider_embeds_requests_with_lineage() -> None:
    fake_model = FakeSentenceTransformer(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    provider = HuggingFaceEmbeddingProvider(
        model_name="test-model",
        dimensions=3,
        max_batch_size=8,
        model=fake_model,
    )
    requests = [
        EmbeddingRequest(
            record_id="chunk_1",
            text="Python is a programming language.",
            metadata={"document_id": "doc_1"},
        ),
        EmbeddingRequest(
            record_id="chunk_2",
            text="BM25 is a keyword retrieval algorithm.",
            metadata={"document_id": "doc_2"},
        ),
    ]

    results = provider.embed_texts(requests)

    assert fake_model.calls == [
        {
            "sentences": [
                "Python is a programming language.",
                "BM25 is a keyword retrieval algorithm.",
            ],
            "batch_size": 2,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
    ]
    assert len(results) == 2
    assert results[0].record_id == "chunk_1"
    assert results[0].vector == [0.1, 0.2, 0.3]
    assert results[0].provider == "huggingface"
    assert results[0].model == "test-model"
    assert results[0].dimensions == 3
    assert results[0].metadata["document_id"] == "doc_1"


def test_huggingface_provider_uses_configured_batch_limit() -> None:
    fake_model = FakeSentenceTransformer(
        embeddings=[
            [0.1],
            [0.2],
            [0.3],
        ]
    )
    provider = HuggingFaceEmbeddingProvider(
        model_name="test-model",
        dimensions=1,
        max_batch_size=2,
        model=fake_model,
    )
    requests = [
        EmbeddingRequest(record_id="chunk_1", text="one"),
        EmbeddingRequest(record_id="chunk_2", text="two"),
        EmbeddingRequest(record_id="chunk_3", text="three"),
    ]

    provider.embed_texts(requests)

    assert fake_model.calls[0]["batch_size"] == 2


def test_huggingface_provider_returns_empty_list_for_empty_requests() -> None:
    fake_model = FakeSentenceTransformer(embeddings=[])
    provider = HuggingFaceEmbeddingProvider(model=fake_model)

    results = provider.embed_texts([])

    assert results == []
    assert fake_model.calls == []


def test_huggingface_provider_rejects_dimension_mismatch() -> None:
    fake_model = FakeSentenceTransformer(embeddings=[[0.1, 0.2]])
    provider = HuggingFaceEmbeddingProvider(
        model_name="test-model",
        dimensions=3,
        model=fake_model,
    )

    with pytest.raises(ValueError, match="expected embedding dimension 3, got 2"):
        provider.embed_texts(
            [
                EmbeddingRequest(
                    record_id="chunk_1",
                    text="Python is a programming language.",
                )
            ]
        )


def test_huggingface_provider_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        HuggingFaceEmbeddingProvider(model_name="")

    with pytest.raises(ValueError, match="dimensions must be greater than 0"):
        HuggingFaceEmbeddingProvider(dimensions=0)

    with pytest.raises(ValueError, match="max_batch_size must be greater than 0"):
        HuggingFaceEmbeddingProvider(max_batch_size=0)


def test_huggingface_provider_retries_after_closed_http_client() -> None:
    fake_model = FakeSentenceTransformer(embeddings=[[0.1, 0.2, 0.3]])
    loader = FailingOnceSentenceTransformerLoader(fake_model)
    provider = HuggingFaceEmbeddingProvider(
        model_name="test-model",
        dimensions=3,
        model_loader=loader,
    )

    results = provider.embed_texts(
        [
            EmbeddingRequest(
                record_id="chunk_1",
                text="Python is a programming language.",
            )
        ]
    )

    assert loader.calls == 2
    assert results[0].vector == [0.1, 0.2, 0.3]
