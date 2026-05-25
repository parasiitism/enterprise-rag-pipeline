from __future__ import annotations

from enterprise_rag.documents import CanonicalDocument
from enterprise_rag.wikipedia.dump_reader import WikipediaPage


def wikipedia_page_to_document(page: WikipediaPage) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=f"wikipedia_{page.page_id}",
        source="wikipedia",
        source_id=page.page_id,
        title=page.title,
        version=page.revision_id,
        updated_at=page.timestamp,
        text=page.text,
        metadata={
            "page_id": page.page_id,
            "revision_id": page.revision_id,
        },
    )
