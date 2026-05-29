from __future__ import annotations

from langchain_core.documents import Document

from enterprise_rag.documents import CanonicalDocument


def canonical_document_to_langchain(document: CanonicalDocument) -> Document:
    return Document(
        page_content=document.text,
        metadata={
            "document_id": document.document_id,
            "source": document.source,
            "source_id": document.source_id,
            "title": document.title,
            "version": document.version,
            "updated_at": document.updated_at,
            **document.metadata,
        },
    )
