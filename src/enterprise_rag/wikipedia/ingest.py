from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_rag.wikipedia.converter import wikipedia_page_to_document
from enterprise_rag.wikipedia.dump_reader import iter_wikipedia_pages


def ingest_wikipedia_dump(
    dump_path: Path,
    output_dir: Path,
    manifest_path: Path,
    limit: int | None = None,
) -> int:
    saved_count = 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for page in iter_wikipedia_pages(dump_path, limit=limit):
            document = wikipedia_page_to_document(page)
            output_path = document.save_json(output_dir)

            manifest_record = {
                "document_id": document.document_id,
                "source": document.source,
                "source_id": document.source_id,
                "title": document.title,
                "version": document.version,
                "updated_at": document.updated_at,
                "output_path": str(output_path),
                "status": "saved",
            }

            manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            saved_count += 1
            print(f"Saved: {document.document_id} | {document.title}")

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia dump into canonical JSON files")

    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to Wikipedia .xml.bz2 dump file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/canonical/wikipedia"),
        help="Directory where canonical JSON files will be saved",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/wikipedia_manifest.jsonl"),
        help="Path where ingestion manifest JSONL will be written",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of Wikipedia pages to ingest",
    )

    args = parser.parse_args()

    print("Starting Wikipedia ingestion")
    print("Dump path:", args.dump_path)
    print("Output dir:", args.output_dir)
    print("Manifest path:", args.manifest_path)
    print("Limit:", args.limit)

    saved_count = ingest_wikipedia_dump(
        dump_path=args.dump_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        limit=args.limit,
    )

    print(f"Done. Saved {saved_count} documents.")


if __name__ == "__main__":
    main()