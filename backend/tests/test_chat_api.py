from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.config import FRONTEND_ORIGIN, RAG_TOP_K
from app.dependencies import get_rag_service
from app.main import app
from app.rag.embeddings import EmbeddingError
from app.rag.llm import LLMError
from app.rag.rag_service import RAGResponse, RAGService
from app.rag.vector_store import VectorSearchResult


@pytest.fixture
def source() -> VectorSearchResult:
    return VectorSearchResult(
        id="aulas/IA/lab07/index.md::9",
        content="Internal chunk content",
        filename="index.md",
        relative_path="aulas/IA/lab07/index.md",
        section="Retropropagação (Backpropagation)",
        chunk_index=9,
        distance=0.312,
    )


@pytest.fixture
def rag_service(source: VectorSearchResult) -> Mock:
    service = Mock(spec=RAGService)
    service.ask.return_value = RAGResponse(
        answer="Backpropagation adjusts neural network weights.",
        sources=[source],
    )
    return service


@pytest.fixture
def client(rag_service: Mock) -> Iterator[TestClient]:
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_still_returns_success(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_success(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "What is backpropagation?"})

    assert response.status_code == 200


def test_chat_response_contains_answer(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Question"})

    assert response.json()["answer"] == (
        "Backpropagation adjusts neural network weights."
    )


def test_chat_response_contains_sources(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Question"})

    assert len(response.json()["sources"]) == 1


def test_source_metadata_is_serialized(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Question"})

    assert response.json()["sources"][0] == {
        "filename": "index.md",
        "relative_path": "aulas/IA/lab07/index.md",
        "section": "Retropropagação (Backpropagation)",
        "chunk_index": 9,
        "distance": 0.312,
    }


def test_chat_without_top_k_uses_configured_default(
    client: TestClient,
    rag_service: Mock,
) -> None:
    client.post("/chat", json={"message": "Question"})

    rag_service.ask.assert_called_once_with("Question", top_k=RAG_TOP_K)


def test_chat_passes_explicit_top_k(client: TestClient, rag_service: Mock) -> None:
    client.post("/chat", json={"message": "Question", "top_k": 3})

    rag_service.ask.assert_called_once_with("Question", top_k=3)


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": " \n\t"})

    assert response.status_code == 422


def test_chat_rejects_zero_top_k(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Question", "top_k": 0})

    assert response.status_code == 422


def test_chat_rejects_top_k_above_limit(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Question", "top_k": 21})

    assert response.status_code == 422


def test_quota_error_returns_service_unavailable(
    client: TestClient,
    rag_service: Mock,
) -> None:
    rag_service.ask.side_effect = EmbeddingError("429 RESOURCE_EXHAUSTED")

    response = client.post("/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "AI service quota is temporarily unavailable."
    }


def test_configuration_error_returns_service_unavailable(
    client: TestClient,
    rag_service: Mock,
) -> None:
    rag_service.ask.side_effect = LLMError(
        "Google API key is required: sensitive-key"
    )

    response = client.post("/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "AI service is not configured."}


def test_unexpected_error_returns_generic_server_error(
    client: TestClient,
    rag_service: Mock,
) -> None:
    rag_service.ask.side_effect = RuntimeError("database internals")

    response = client.post("/chat", json={"message": "Question"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_error_response_does_not_expose_api_key(
    client: TestClient,
    rag_service: Mock,
) -> None:
    secret = "sensitive-api-key"
    rag_service.ask.side_effect = LLMError(f"request failed with {secret}")

    response = client.post("/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert secret not in response.text


def test_response_does_not_expose_internal_prompt(
    client: TestClient,
    rag_service: Mock,
) -> None:
    internal_prompt = "INTERNAL PROMPT WITH PRIVATE CONTEXT"
    rag_service.ask.side_effect = RuntimeError(internal_prompt)

    response = client.post("/chat", json={"message": "Question"})

    assert internal_prompt not in response.text
    assert "prompt" not in response.text.lower()


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
