from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.rag import embeddings
from app.rag.embeddings import EmbeddedChunk, EmbeddingError, GoogleEmbeddingService
from app.rag.markdown_chunker import MarkdownChunk


def embedding_response(*vectors: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


def mock_client(monkeypatch: pytest.MonkeyPatch, response: object) -> Mock:
    embed_content = Mock(return_value=response)
    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    monkeypatch.setattr(embeddings.genai, "Client", Mock(return_value=client))
    return embed_content


def make_chunk(
    content: str,
    filename: str = "guide.md",
    relative_path: str = "docs/guide.md",
    section: str | None = "Guide",
    chunk_index: int = 0,
) -> MarkdownChunk:
    return MarkdownChunk(
        content=content,
        filename=filename,
        relative_path=relative_path,
        section=section,
        chunk_index=chunk_index,
    )


def test_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client_factory = Mock()
    monkeypatch.setattr(embeddings, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(embeddings.genai, "Client", client_factory)

    with pytest.raises(EmbeddingError, match="Google API key is required"):
        GoogleEmbeddingService()

    client_factory.assert_not_called()


def test_embed_text_rejects_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_content = mock_client(monkeypatch, embedding_response([0.1]))
    service = GoogleEmbeddingService(api_key="test-key")

    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_text(" \n\t")

    embed_content.assert_not_called()


def test_embed_text_returns_list_of_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client(monkeypatch, embedding_response([1, 2.5, -3]))

    vector = GoogleEmbeddingService(api_key="test-key").embed_text("FastAPI")

    assert vector == [1.0, 2.5, -3.0]
    assert all(isinstance(value, float) for value in vector)


def test_embed_text_passes_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_content = mock_client(monkeypatch, embedding_response([0.1]))

    GoogleEmbeddingService(api_key="test-key", model="custom-model").embed_text(
        "Text"
    )

    assert embed_content.call_args.kwargs["model"] == "custom-model"


def test_embed_text_sends_text_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_content = mock_client(monkeypatch, embedding_response([0.1]))

    GoogleEmbeddingService(api_key="test-key").embed_text("Expected text")

    assert embed_content.call_args.kwargs["contents"] == "Expected text"


def test_embed_texts_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_content = mock_client(
        monkeypatch,
        embedding_response([1.0], [2.0], [3.0]),
    )
    texts = ["first", "second", "third"]

    vectors = GoogleEmbeddingService(api_key="test-key").embed_texts(texts)

    assert vectors == [[1.0], [2.0], [3.0]]
    assert embed_content.call_args.kwargs["contents"] == texts


def test_embed_texts_validates_all_texts_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed_content = mock_client(monkeypatch, embedding_response([1.0], [2.0]))
    service = GoogleEmbeddingService(api_key="test-key")

    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_texts(["valid", " "])

    embed_content.assert_not_called()


def test_embed_chunks_returns_one_embedded_chunk_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client(monkeypatch, embedding_response([1.0], [2.0]))
    chunks = [make_chunk("First", chunk_index=0), make_chunk("Second", chunk_index=1)]

    embedded_chunks = GoogleEmbeddingService(api_key="test-key").embed_chunks(chunks)

    assert len(embedded_chunks) == 2
    assert all(isinstance(item, EmbeddedChunk) for item in embedded_chunks)
    assert [item.embedding for item in embedded_chunks] == [[1.0], [2.0]]


def test_embed_chunks_preserves_original_chunk_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client(monkeypatch, embedding_response([0.1, 0.2]))
    chunk = make_chunk(
        "Installation content",
        filename="spring.md",
        relative_path="java/spring.md",
        section="Installation",
        chunk_index=4,
    )

    embedded_chunk = GoogleEmbeddingService(api_key="test-key").embed_chunks([chunk])[0]

    assert embedded_chunk.chunk is chunk
    assert embedded_chunk.chunk.filename == "spring.md"
    assert embedded_chunk.chunk.relative_path == "java/spring.md"
    assert embedded_chunk.chunk.section == "Installation"
    assert embedded_chunk.chunk.chunk_index == 4


def test_invalid_api_response_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client(monkeypatch, SimpleNamespace(embeddings=[]))

    with pytest.raises(EmbeddingError, match="did not contain the expected vectors"):
        GoogleEmbeddingService(api_key="test-key").embed_text("Text")


def test_api_key_is_redacted_from_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-test-key"
    embed_content = mock_client(monkeypatch, embedding_response([0.1]))
    embed_content.side_effect = RuntimeError(f"request failed using {secret}")

    with pytest.raises(EmbeddingError) as error_info:
        GoogleEmbeddingService(api_key=secret).embed_text("Text")

    assert secret not in str(error_info.value)
    assert "[REDACTED]" in str(error_info.value)


def test_embed_texts_returns_empty_list_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed_content = mock_client(monkeypatch, embedding_response([0.1]))

    vectors = GoogleEmbeddingService(api_key="test-key").embed_texts([])

    assert vectors == []
    embed_content.assert_not_called()
