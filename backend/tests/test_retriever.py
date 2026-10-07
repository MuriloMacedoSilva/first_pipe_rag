from unittest.mock import Mock

import pytest

from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


MEANINGFUL_CONTENT = (
    "This chunk contains enough useful information to pass the content filter."
)


def make_result(
    *,
    content: str = MEANINGFUL_CONTENT,
    distance: float | None = 0.4,
    relative_path: str = "docs/guide.md",
    chunk_index: int = 0,
) -> VectorSearchResult:
    return VectorSearchResult(
        id=f"{relative_path}::{chunk_index}",
        content=content,
        filename=relative_path.rsplit("/", 1)[-1],
        relative_path=relative_path,
        section="Introduction",
        chunk_index=chunk_index,
        distance=distance,
    )


@pytest.fixture
def embedding_service() -> Mock:
    service = Mock(spec=GoogleEmbeddingService)
    service.embed_text.return_value = [0.9, 0.1, 0.0]
    return service


@pytest.fixture
def vector_store() -> Mock:
    store = Mock(spec=ChromaVectorStore)
    store.count.return_value = 10
    store.query.return_value = []
    return store


def test_valid_query_generates_embedding(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    RAGRetriever(embedding_service, vector_store).retrieve("How does it work?")

    embedding_service.embed_text.assert_called_once_with("How does it work?")


def test_query_embedding_is_sent_to_vector_store(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    embedding_service.embed_text.return_value = [0.2, 0.8]

    RAGRetriever(embedding_service, vector_store).retrieve("Question")

    vector_store.query.assert_called_once_with(
        embedding=[0.2, 0.8],
        n_results=10,
    )


def test_result_below_distance_threshold_is_kept(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result(distance=0.84)
    vector_store.query.return_value = [result]

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == [result]


def test_result_above_distance_threshold_is_discarded(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.query.return_value = [make_result(distance=0.86)]

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == []


def test_result_at_distance_threshold_is_kept(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result(distance=0.85)
    vector_store.query.return_value = [result]

    assert RAGRetriever(embedding_service, vector_store).retrieve("Question") == [
        result
    ]


def test_result_without_distance_is_kept(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result(distance=None)
    vector_store.query.return_value = [result]

    assert RAGRetriever(embedding_service, vector_store).retrieve("Question") == [
        result
    ]


def test_header_only_chunk_is_discarded(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.query.return_value = [
        make_result(content="## Modern Activation Functions")
    ]

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == []


def test_header_with_enough_content_is_kept(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result(
        content=(
            "## Modern Activation Functions\n\n"
            "ReLU returns zero for negative values and preserves positive values."
        )
    )
    vector_store.query.return_value = [result]

    assert RAGRetriever(embedding_service, vector_store).retrieve("Question") == [
        result
    ]


def test_useful_code_chunk_is_kept(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result(content="```python\ndef backward():\n    ...\n```")
    vector_store.query.return_value = [result]

    assert RAGRetriever(embedding_service, vector_store).retrieve("Question") == [
        result
    ]


def test_query_requests_more_candidates_than_top_k(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.count.return_value = 100

    RAGRetriever(
        embedding_service,
        vector_store,
        candidate_multiplier=3,
    ).retrieve("Question", top_k=4)

    assert vector_store.query.call_args.kwargs["n_results"] == 12


def test_final_results_never_exceed_top_k(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.query.return_value = [
        make_result(chunk_index=index) for index in range(6)
    ]

    results = RAGRetriever(embedding_service, vector_store).retrieve(
        "Question", top_k=3
    )

    assert len(results) == 3


def test_candidate_count_does_not_exceed_collection_count(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.count.return_value = 7

    RAGRetriever(embedding_service, vector_store).retrieve("Question", top_k=5)

    assert vector_store.query.call_args.kwargs["n_results"] == 7


def test_all_filtered_results_return_empty_list(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    vector_store.query.return_value = [
        make_result(distance=0.9),
        make_result(content="# Header", chunk_index=1),
    ]

    assert RAGRetriever(embedding_service, vector_store).retrieve("Question") == []


def test_filters_preserve_original_ranking_order(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    first = make_result(distance=0.2, chunk_index=0)
    filtered = make_result(distance=0.9, chunk_index=1)
    second = make_result(distance=0.4, chunk_index=2)
    vector_store.query.return_value = [first, filtered, second]

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == [first, second]


def test_duplicate_chunk_is_returned_only_once(
    embedding_service: Mock,
    vector_store: Mock,
) -> None:
    result = make_result()
    vector_store.query.return_value = [result, result]

    results = RAGRetriever(embedding_service, vector_store).retrieve("Question")

    assert results == [result]


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
