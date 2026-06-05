import json

from enterprise_rag.embedding_store import (
    append_embedding_manifest_record,
    embedding_result_to_record,
    save_embedding_record,
)
from enterprise_rag.embeddings import EmbeddingResult


def test_embedding_result_to_record_preserves_lineage() -> None:
    result = EmbeddingResult(
        record_id="chunk:1",
        vector=[0.1, 0.2, 0.3],
        provider="huggingface",
        model="test-model",
        dimensions=3,
        metadata={
            "chunk_id": "chunk:1",
            "document_id": "doc_1",
            "title": "Test Document",
        },
    )

    record = embedding_result_to_record(result)

    assert record["record_id"] == "chunk:1"
    assert record["chunk_id"] == "chunk:1"
    assert record["document_id"] == "doc_1"
    assert record["title"] == "Test Document"
    assert record["provider"] == "huggingface"
    assert record["model"] == "test-model"
    assert record["dimensions"] == 3
    assert record["vector"] == [0.1, 0.2, 0.3]


def test_save_embedding_record_sanitizes_record_id(tmp_path) -> None:
    record = {
        "record_id": "wiki:doc/chunk\\1",
        "vector": [0.1],
    }

    output_path = save_embedding_record(record, tmp_path)

    assert output_path.name == "wiki_doc_chunk_1.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == record


def test_append_embedding_manifest_record_writes_jsonl(tmp_path) -> None:
    record = {
        "record_id": "chunk_1",
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "provider": "huggingface",
        "model": "test-model",
        "dimensions": 3,
    }
    output_path = tmp_path / "embeddings" / "chunk_1.json"
    manifest_path = tmp_path / "manifests" / "embedding_manifest.jsonl"

    append_embedding_manifest_record(
        record=record,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    manifest_record = json.loads(lines[0])
    assert manifest_record["record_id"] == "chunk_1"
    assert manifest_record["output_path"] == str(output_path)
