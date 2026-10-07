from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.config import CHROMA_COLLECTION, CHROMA_PATH
from app.rag.embeddings import EmbeddedChunk
from app.rag.markdown_chunker import MarkdownChunk


@dataclass
class VectorSearchResult:
    id: str
    content: str
    filename: str
    relative_path: str
    section: str | None
    chunk_index: int
    distance: float | None


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        configured_path = persist_path if persist_path is not None else CHROMA_PATH
        self.persist_path = Path(configured_path).expanduser().resolve()
        self.collection_name = collection_name or CHROMA_COLLECTION
        self._client = chromadb.PersistentClient(path=self.persist_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(self, chunks: Iterable[EmbeddedChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return

        self._collection.upsert(
            ids=[self.make_chunk_id(item.chunk) for item in chunk_list],
            documents=[item.chunk.content for item in chunk_list],
            embeddings=[item.embedding for item in chunk_list],
            metadatas=[self._metadata(item) for item in chunk_list],
        )

    @staticmethod
    def make_chunk_id(chunk: MarkdownChunk) -> str:
        return f"{chunk.relative_path}::{chunk.chunk_index}"

    def contains_id(self, chunk_id: str) -> bool:
        return chunk_id in self.existing_ids([chunk_id])

    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        id_list = list(dict.fromkeys(ids))
        if not id_list:
            return set()

        response = self._collection.get(ids=id_list, include=[])
        return set(response["ids"])

    def query(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[VectorSearchResult]:
        if not embedding:
            raise ValueError("Query embedding cannot be empty")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        stored_count = self.count()
        if stored_count == 0:
            return []

        response = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, stored_count),
            include=["documents", "metadatas", "distances"],
        )

        ids = self._first_batch(response.get("ids"))
        documents = self._first_batch(response.get("documents"))
        metadatas = self._first_batch(response.get("metadatas"))
        distances = self._first_batch(response.get("distances"))

        results: list[VectorSearchResult] = []
        for index, item_id in enumerate(ids):
            metadata = metadatas[index] or {}
            section = metadata.get("section")
            distance = distances[index] if index < len(distances) else None
            results.append(
                VectorSearchResult(
                    id=item_id,
                    content=documents[index] or "",
                    filename=str(metadata.get("filename", "")),
                    relative_path=str(metadata.get("relative_path", "")),
                    section=str(section) if section else None,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return results

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    @staticmethod
    def _metadata(embedded_chunk: EmbeddedChunk) -> dict[str, str | int]:
        chunk = embedded_chunk.chunk
        return {
            "filename": chunk.filename,
            "relative_path": chunk.relative_path,
            "section": chunk.section or "",
            "chunk_index": chunk.chunk_index,
        }

    @staticmethod
    def _first_batch(value: Any) -> list[Any]:
        return list(value[0]) if value else []
