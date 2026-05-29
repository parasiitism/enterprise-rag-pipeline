from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def chunk_to_record(chunk: Document) -> dict[str, object]:
    return {
        "chunk_id": chunk.metadata["chunk_id"],
        "document_id": chunk.metadata["document_id"],
        "source": chunk.metadata["source"],
        "source_id": chunk.metadata["source_id"],
        "title": chunk.metadata["title"],
        "chunk_index": chunk.metadata["chunk_index"],
        "text": chunk.page_content,
        "metadata": chunk.metadata,
    }


def save_chunk_record(record: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = str(record["chunk_id"])
    output_path = output_dir / f"{chunk_id}.json"

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path