from pathlib import Path

import pytest

from app.rag.embeddings import EmbeddedChunk
from app.rag.markdown_chunker import MarkdownChunk
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_path=tmp_path / "chroma",
        collection_name="test_collection",
    )


def make_embedded_chunk(
    content: str = "Content",
    embedding: list[float] | None = None,
    filename: str = "document.md",
    relative_path: str = "docs/document.md",
    section: str | None = "Section",
    chunk_index: int = 0,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=MarkdownChunk(
            content=content,
            filename=filename,
            relative_path=relative_path,
            section=section,
            chunk_index=chunk_index,
        ),
        embedding=embedding or [1.0, 0.0, 0.0],
    )


def test_creates_empty_collection(store: ChromaVectorStore) -> None:
    assert store.collection_name == "test_collection"
    assert store.persist_path.exists()


def test_count_is_zero_initially(store: ChromaVectorStore) -> None:
    assert store.count() == 0


def test_adds_one_embedded_chunk(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk()])

    assert store.count() == 1


def test_adds_multiple_embedded_chunks(store: ChromaVectorStore) -> None:
    chunks = [
        make_embedded_chunk(relative_path="a.md", chunk_index=0),
        make_embedded_chunk(relative_path="a.md", chunk_index=1),
        make_embedded_chunk(relative_path="b.md", chunk_index=0),
    ]

    store.add_chunks(chunks)

    assert store.count() == 3


def test_preserves_document_and_metadata(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [
            make_embedded_chunk(
                content="Spring content",
                filename="spring.md",
                relative_path="java/spring.md",
                section="Installation",
                chunk_index=3,
            )
        ]
    )

    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.content == "Spring content"
    assert result.filename == "spring.md"
    assert result.relative_path == "java/spring.md"
    assert result.section == "Installation"
    assert result.chunk_index == 3


def test_count_returns_number_of_stored_chunks(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [
            make_embedded_chunk(relative_path="one.md"),
            make_embedded_chunk(relative_path="two.md"),
        ]
    )

    assert store.count() == 2


def test_reindexing_same_id_does_not_duplicate_record(
    store: ChromaVectorStore,
) -> None:
    store.add_chunks([make_embedded_chunk(content="Old content")])
    store.add_chunks([make_embedded_chunk(content="Updated content")])

    assert store.count() == 1
    assert store.query([1.0, 0.0, 0.0], n_results=1)[0].content == "Updated content"


def test_query_orders_results_by_vector_similarity(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [
            make_embedded_chunk(
                content="Document A",
                embedding=[1.0, 0.0, 0.0],
                relative_path="a.md",
            ),
            make_embedded_chunk(
                content="Document B",
                embedding=[0.0, 1.0, 0.0],
                relative_path="b.md",
            ),
        ]
    )

    results = store.query([0.9, 0.1, 0.0], n_results=2)

    assert [result.content for result in results] == ["Document A", "Document B"]


def test_query_preserves_relative_path(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk(relative_path="cloud/gcp.md")])

    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.relative_path == "cloud/gcp.md"


def test_query_preserves_section(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk(section="Configuration")])

    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.section == "Configuration"


def test_query_restores_missing_section_as_none(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk(section=None)])

    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.section is None


def test_query_preserves_chunk_index(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk(chunk_index=7)])

    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.chunk_index == 7


def test_query_returns_distance(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk()])

    result = store.query([0.9, 0.1, 0.0], n_results=1)[0]

    assert isinstance(result, VectorSearchResult)
    assert isinstance(result.distance, float)


def test_query_rejects_empty_embedding(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="Query embedding cannot be empty"):
        store.query([])


def test_query_rejects_non_positive_result_count(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="n_results must be greater than zero"):
        store.query([1.0, 0.0], n_results=0)

    with pytest.raises(ValueError, match="n_results must be greater than zero"):
        store.query([1.0, 0.0], n_results=-1)


def test_clear_empties_and_recreates_collection(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk()])

    store.clear()

    assert store.count() == 0
    store.add_chunks([make_embedded_chunk()])
    assert store.count() == 1


def test_collection_persists_between_store_instances(tmp_path: Path) -> None:
    persist_path = tmp_path / "persistent_chroma"
    first_store = ChromaVectorStore(persist_path, "persistent_collection")
    first_store.add_chunks([make_embedded_chunk()])

    second_store = ChromaVectorStore(persist_path, "persistent_collection")

    assert second_store.count() == 1


def test_add_empty_collection_does_nothing(store: ChromaVectorStore) -> None:
    store.add_chunks([])

    assert store.count() == 0


def test_chunk_id_is_deterministic(store: ChromaVectorStore) -> None:
    chunk = make_embedded_chunk(relative_path="java/spring.md", chunk_index=2)

    store.add_chunks([chunk])
    result = store.query([1.0, 0.0, 0.0], n_results=1)[0]

    assert result.id == "java/spring.md::2"
    assert store.make_chunk_id(chunk.chunk) == "java/spring.md::2"


def test_existing_ids_returns_only_stored_ids(store: ChromaVectorStore) -> None:
    stored = make_embedded_chunk(relative_path="stored.md", chunk_index=0)
    store.add_chunks([stored])

    existing = store.existing_ids(
        ["stored.md::0", "missing.md::0", "stored.md::0"]
    )

    assert existing == {"stored.md::0"}


def test_contains_id_detects_existing_chunk(store: ChromaVectorStore) -> None:
    store.add_chunks([make_embedded_chunk(relative_path="stored.md")])

    assert store.contains_id("stored.md::0") is True
    assert store.contains_id("missing.md::0") is False
