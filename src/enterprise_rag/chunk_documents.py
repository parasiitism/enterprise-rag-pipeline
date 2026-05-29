from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.chunk_store import chunk_to_record, save_chunk_record
from enterprise_rag.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_canonical_document
from enterprise_rag.document_store import iter_canonical_documents


@dataclass(frozen=True)
class ChunkingSummary:
    documents_processed: int
    chunks_saved: int


def chunk_document_directory(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    limit: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingSummary:
    documents_processed = 0
    chunks_saved = 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for document in iter_canonical_documents(input_dir=input_dir, limit=limit):
            chunks = chunk_canonical_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            documents_processed += 1

            for chunk in chunks:
                record = chunk_to_record(chunk)
                output_path = save_chunk_record(record, output_dir)

                manifest_file.write(
                    json.dumps(
                        {
                            "document_id": record["document_id"],
                            "chunk_id": record["chunk_id"],
                            "output_path": str(output_path),
                            "status": "saved",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunks_saved += 1

    return ChunkingSummary(
        documents_processed=documents_processed,
        chunks_saved=chunks_saved,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk canonical documents")

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks/wikipedia"))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/chunk_manifest.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    args = parser.parse_args()

    summary = chunk_document_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Documents processed: {summary.documents_processed}")
    print(f"Chunks saved: {summary.chunks_saved}")


if __name__ == "__main__":
    main()