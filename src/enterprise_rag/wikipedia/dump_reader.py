from __future__ import annotations

import bz2
import xml.etree.ElementTree as ET
import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WikipediaPage:
    title: str
    page_id: str
    revision_id: str
    timestamp: str
    text: str


def iter_wikipedia_pages(dump_path: Path, limit: int | None = None) -> Iterator[WikipediaPage]:
    count = 0

    with bz2.open(dump_path, mode="rb") as file:
        context = ET.iterparse(file, events=("end",))

        print(f"context:{context}")
        for _, elem in context:
            if not elem.tag.endswith("page"):
                continue

            page = _parse_page(elem)
            elem.clear()

            if page is None:
                continue

            yield page
            count += 1

            if limit is not None and count >= limit:
                break


def _parse_page(page_elem: ET.Element) -> WikipediaPage | None:
    title = _find_text(page_elem, "title")
    page_id = _find_direct_child_text(page_elem, "id")
    revision = _find_child(page_elem, "revision")

    if revision is None:
        return None

    revision_id = _find_direct_child_text(revision, "id")
    timestamp = _find_text(revision, "timestamp")
    text = _find_text(revision, "text")

    if not title or not page_id or not revision_id:
        return None

    return WikipediaPage(
        title=title,
        page_id=page_id,
        revision_id=revision_id,
        timestamp=timestamp or "",
        text=text or "",
    )


def _find_text(elem: ET.Element, name: str) -> str | None:
    child = _find_child(elem, name)
    if child is None:
        return None
    return child.text


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem.iter():
        if child.tag.endswith(name):
            return child
    return None


def _find_direct_child_text(elem: ET.Element, name: str) -> str | None:
    for child in elem:
        if child.tag.endswith(name):
            return child.text
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Wikipedia dump reader")

    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to Wikipedia .xml.bz2 dump file",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of pages to read for debugging",
    )

    args = parser.parse_args()

    print("Starting Wikipedia dump debug")
    print("Dump path:", args.dump_path)
    print("Limit:", args.limit)

    for index, page in enumerate(
        iter_wikipedia_pages(args.dump_path, limit=args.limit),
        start=1,
    ):
        print("\n==============================")
        print("Text preview:", page.text)
if __name__ == "__main__":
    main()
