from unittest.mock import Mock

import pytest

from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


@pytest.fixture
def embedding_service() -> Mock:
    service = Mock(spec=GoogleEmbeddingService)
    service.embed_text.return_value = [0.9, 0.1, 0.0]
    return service


@pytest.fixture
def vector_store() -> Mock:
    store = Mock(spec=ChromaVectorStore)
    store.count.return_value = 3
    store.query.return_value = []
    return store


@pytest.fixture
def search_results() -> list[VectorSearchResult]:
    return [
        VectorSearchResult(
            id="docs/guide.md::0",
            content="Relevant content",
            filename="guide.md",
            relative_path="docs/guide.md",
            section="Introduction",
            chunk_index=0,
            distance=0.1,
        )
    ]


def test_valid_query_generates_embedding(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    retriever = RAGRetriever(embedding_service, vector_store)

    retriever.retrieve("How does it work?")

    embedding_service.embed_text.assert_called_once_with("How does it work?")


def test_query_embedding_is_sent_to_vector_store(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    embedding_service.embed_text.return_value = [0.2, 0.8]

    RAGRetriever(embedding_service, vector_store).retrieve("Question")

    vector_store.query.assert_called_once_with(
        embedding=[0.2, 0.8],
        n_results=5,
    )


def test_top_k_is_passed_to_vector_store(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    RAGRetriever(embedding_service, vector_store).retrieve("Question", top_k=10)

    assert vector_store.query.call_args.kwargs["n_results"] == 10


def test_vector_store_results_are_returned_unchanged(
    embedding_service: Mock,
    vector_store: Mock,
    search_results: list[VectorSearchResult],
) -> None:
    vector_store.query.return_value = search_results

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results is search_results


def test_empty_query_raises_error(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        RAGRetriever(embedding_service, vector_store).retrieve("")


def test_whitespace_query_raises_error(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        RAGRetriever(embedding_service, vector_store).retrieve(" \n\t")


def test_non_positive_top_k_raises_error(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    retriever = RAGRetriever(embedding_service, vector_store)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.retrieve("Question", top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.retrieve("Question", top_k=-1)


def test_empty_collection_returns_empty_list(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.count.return_value = 0

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == []


def test_empty_collection_does_not_generate_embedding(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.count.return_value = 0

    RAGRetriever(embedding_service, vector_store).retrieve("Question")

    embedding_service.embed_text.assert_not_called()
    vector_store.query.assert_not_called()


def test_embedding_error_is_propagated(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    embedding_service.embed_text.side_effect = EmbeddingError("Embedding failed")

    with pytest.raises(EmbeddingError, match="Embedding failed"):
        RAGRetriever(embedding_service, vector_store).retrieve("Question")


def test_retrieval_never_writes_to_vector_store(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    RAGRetriever(embedding_service, vector_store).retrieve("Question")

    vector_store.add_chunks.assert_not_called()
    vector_store.clear.assert_not_called()
