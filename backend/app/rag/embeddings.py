from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from google import genai

from app.config import GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL
from app.rag.markdown_chunker import MarkdownChunk


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddedChunk:
    chunk: MarkdownChunk
    embedding: list[float]


class GoogleEmbeddingService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else GOOGLE_API_KEY
        if not resolved_api_key or not resolved_api_key.strip():
            raise EmbeddingError("Google API key is required for embeddings")

        resolved_model = model if model is not None else GOOGLE_EMBEDDING_MODEL
        if not resolved_model or not resolved_model.strip():
            raise EmbeddingError("Google embedding model is required")

        self._api_key = resolved_api_key.strip()
        self.model = resolved_model.strip()
        self._client = genai.Client(api_key=self._api_key)

    def embed_text(self, text: str) -> list[float]:
        self._validate_text(text)
        return self._request_embeddings(text, expected_count=1)[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        for text in text_list:
            self._validate_text(text)

        if not text_list:
            return []

        return self._request_embeddings(text_list, expected_count=len(text_list))

    def embed_chunks(self, chunks: Iterable[MarkdownChunk]) -> list[EmbeddedChunk]:
        chunk_list = list(chunks)
        embeddings = self.embed_texts(chunk.content for chunk in chunk_list)
        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunk_list, embeddings, strict=True)
        ]

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to embed cannot be empty")

    def _request_embeddings(
        self,
        contents: str | list[str],
        expected_count: int,
    ) -> list[list[float]]:
        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=contents,
            )
        except Exception as error:
            self._raise_request_error(error)

        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != expected_count:
            raise EmbeddingError(
                "Google embedding response did not contain the expected vectors"
            )

        return [self._extract_vector(embedding) for embedding in embeddings]

    @staticmethod
    def _extract_vector(embedding: Any) -> list[float]:
        values = getattr(embedding, "values", None)
        if not values:
            raise EmbeddingError(
                "Google embedding response contained an empty or invalid vector"
            )

        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise EmbeddingError(
                "Google embedding response contained non-numeric vector values"
            ) from error

    def _raise_request_error(self, error: Exception) -> None:
        details = str(error).replace(self._api_key, "[REDACTED]")
        message = f"Google embedding request failed ({type(error).__name__})"
        if details:
            message = f"{message}: {details}"
        raise EmbeddingError(message) from None
