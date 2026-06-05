from __future__ import annotations

import json
from pathlib import Path

from enterprise_rag.embeddings.models import EmbeddingResult


def embedding_result_to_record(result: EmbeddingResult) -> dict[str, object]:
    metadata = dict(result.metadata)

    return {
        "record_id": result.record_id,
        "chunk_id": metadata.get("chunk_id", result.record_id),
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title"),
        "provider": result.provider,
        "model": result.model,
        "dimensions": result.dimensions,
        "vector": result.vector,
        "metadata": metadata,
    }


def save_embedding_record(record: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    record_id = str(record["record_id"])
    safe_record_id = record_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_record_id}.json"

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def append_embedding_manifest_record(
    record: dict[str, object],
    output_path: Path,
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_record = {
        "record_id": record["record_id"],
        "chunk_id": record["chunk_id"],
        "document_id": record["document_id"],
        "provider": record["provider"],
        "model": record["model"],
        "dimensions": record["dimensions"],
        "output_path": str(output_path),
    }

    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")