from __future__ import annotations

from rank_bm25 import BM25Okapi

from enterprise_rag.retrieval.models import SearchResult
from enterprise_rag.retrieval.tokenizer import tokenize


class BM25ChunkIndex:
    def __init__(self, chunk_records: list[dict[str, object]]) -> None:
        if not chunk_records:
            raise ValueError("chunk_records cannot be empty")

        self.chunk_records = chunk_records
        self.tokenized_corpus = [
            tokenize(str(record["text"]))
            for record in chunk_records
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        scores = self.index.get_scores(query_tokens)

        matching_indexes = [
            index
            for index, document_tokens in enumerate(self.tokenized_corpus)
            if query_token_set.intersection(document_tokens)
        ]

        ranked_indexes = sorted(
            matching_indexes,
            key=lambda index: (
                scores[index],
                len(query_token_set.intersection(self.tokenized_corpus[index])),
            ),
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []

        for index in ranked_indexes:
            record = self.chunk_records[index]
            score = float(scores[index])

            results.append(
                SearchResult(
                    chunk_id=str(record["chunk_id"]),
                    document_id=str(record["document_id"]),
                    title=str(record["title"]),
                    score=score,
                    text=str(record["text"]),
                    metadata=dict(record["metadata"]),
                )
            )

        return results
