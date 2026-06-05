import json

import pytest

from enterprise_rag.embed_chunks import (
    chunk_record_to_embedding_request,
    embed_chunk_records,
    iter_batches,
)
from enterprise_rag.embeddings import EmbeddingProviderInfo, EmbeddingRequest, EmbeddingResult


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.info = EmbeddingProviderInfo(
            provider="fake",
            model="fake-model",
            dimensions=3,
            max_batch_size=2,
        )
        self.requests: list[list[EmbeddingRequest]] = []

    def embed_texts(
        self,
        requests: list[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        self.requests.append(requests)

        return [
            EmbeddingResult(
                record_id=request.record_id,
                vector=[0.1, 0.2, 0.3],
                provider=self.info.provider,
                model=self.info.model,
                dimensions=self.info.dimensions,
                metadata=request.metadata,
            )
            for request in requests
        ]


def _chunk_record(chunk_id: str, text: str = "Python text") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_1",
        "source": "wikipedia",
        "source_id": "1",
        "title": "Python",
        "chunk_index": 0,
        "text": text,
        "metadata": {"existing": "metadata"},
    }


def test_chunk_record_to_embedding_request_preserves_metadata() -> None:
    request = chunk_record_to_embedding_request(_chunk_record("chunk_1"))

    assert request.record_id == "chunk_1"
    assert request.text == "Python text"
    assert request.metadata["chunk_id"] == "chunk_1"
    assert request.metadata["document_id"] == "doc_1"
    assert request.metadata["existing"] == "metadata"


def test_iter_batches_yields_fixed_size_batches() -> None:
    batches = list(iter_batches([{"id": 1}, {"id": 2}, {"id": 3}], batch_size=2))

    assert batches == [[{"id": 1}, {"id": 2}], [{"id": 3}]]


def test_iter_batches_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        list(iter_batches([], batch_size=0))


def test_embed_chunk_records_writes_embeddings_and_manifest(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    records = [
        _chunk_record("chunk_1", text="first chunk"),
        _chunk_record("chunk_2", text="second chunk"),
        _chunk_record("chunk_3", text="third chunk"),
    ]
    output_dir = tmp_path / "embeddings"
    manifest_path = tmp_path / "manifests" / "embedding_manifest.jsonl"

    total = embed_chunk_records(
        records=records,
        provider=provider,
        output_dir=output_dir,
        manifest_path=manifest_path,
        batch_size=2,
    )

    assert total == 3
    assert len(provider.requests) == 2
    assert (output_dir / "chunk_1.json").exists()
    assert (output_dir / "chunk_2.json").exists()
    assert (output_dir / "chunk_3.json").exists()

    saved_record = json.loads((output_dir / "chunk_1.json").read_text(encoding="utf-8"))
    assert saved_record["record_id"] == "chunk_1"
    assert saved_record["document_id"] == "doc_1"
    assert saved_record["vector"] == [0.1, 0.2, 0.3]

    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 3
