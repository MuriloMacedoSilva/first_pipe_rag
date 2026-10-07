import re

from app.config import (
    RAG_CANDIDATE_MULTIPLIER,
    RAG_MAX_DISTANCE,
    RAG_MIN_CONTENT_CHARS,
)
from app.rag.embeddings import GoogleEmbeddingService
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


_MARKDOWN_HEADER_PATTERN = re.compile(r"^\s{0,3}#{1,6}(?:\s+.*)?\s*$")
_FENCED_CODE_PATTERN = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
_CODE_LINE_PATTERN = re.compile(
    r"^\s*(?:async\s+def|def|class|from|import|for|while|if)\b",
    re.MULTILINE,
)


class RAGRetriever:
    def __init__(
        self,
        embedding_service: GoogleEmbeddingService,
        vector_store: ChromaVectorStore,
        max_distance: float = RAG_MAX_DISTANCE,
        candidate_multiplier: int = RAG_CANDIDATE_MULTIPLIER,
        min_content_chars: int = RAG_MIN_CONTENT_CHARS,
    ) -> None:
        if max_distance < 0:
            raise ValueError("max_distance cannot be negative")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if min_content_chars <= 0:
            raise ValueError("min_content_chars must be greater than zero")

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.max_distance = max_distance
        self.candidate_multiplier = candidate_multiplier
        self.min_content_chars = min_content_chars

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection_count = self.vector_store.count()
        if collection_count == 0:
            return []

        candidate_count = min(
            max(top_k, top_k * self.candidate_multiplier),
            collection_count,
        )
        query_embedding = self.embedding_service.embed_text(query)
        candidates = self.vector_store.query(
            embedding=query_embedding,
            n_results=candidate_count,
        )
        return self._filter_results(candidates, top_k)

    def _filter_results(
        self,
        results: list[VectorSearchResult],
        top_k: int,
    ) -> list[VectorSearchResult]:
        filtered: list[VectorSearchResult] = []
        seen_chunks: set[tuple[str, int]] = set()

        for result in results:
            if result.distance is not None and result.distance > self.max_distance:
                continue
            if not self._has_meaningful_content(result.content):
                continue

            chunk_key = (result.relative_path, result.chunk_index)
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            filtered.append(result)

            if len(filtered) == top_k:
                break

        return filtered

    def _has_meaningful_content(self, content: str) -> bool:
        without_headers = "\n".join(
            line
            for line in content.splitlines()
            if not _MARKDOWN_HEADER_PATTERN.fullmatch(line)
        ).strip()
        normalized_content = " ".join(without_headers.split())
        if len(normalized_content) >= self.min_content_chars:
            return True

        fenced_code = _FENCED_CODE_PATTERN.findall(without_headers)
        if any(len(" ".join(code.split())) >= 10 for code in fenced_code):
            return True

        return (
            len(normalized_content) >= 10
            and _CODE_LINE_PATTERN.search(without_headers) is not None
        )
