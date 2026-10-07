from unittest.mock import Mock

import pytest

from app.rag.embeddings import EmbeddedChunk, EmbeddingError
from app.rag.markdown_chunker import MarkdownChunk
from scripts.index_material import index_chunks, parse_args


def make_chunks(count: int) -> list[MarkdownChunk]:
    return [
        MarkdownChunk(
            content=f"Content {index}",
            filename="document.md",
            relative_path="docs/document.md",
            section="Section",
            chunk_index=index,
        )
        for index in range(count)
    ]


class FakeStore:
    collection_name = "test_collection"

    def __init__(
        self,
        existing_ids: set[str] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.ids = set(existing_ids or set())
        self.events = events
        self.clear_called = False
        self.existing_ids_calls = 0

    @staticmethod
    def make_chunk_id(chunk: MarkdownChunk) -> str:
        return f"{chunk.relative_path}::{chunk.chunk_index}"

    def existing_ids(self, ids: list[str]) -> set[str]:
        self.existing_ids_calls += 1
        return self.ids.intersection(ids)

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        if self.events is not None:
            self.events.append("persist")
        self.ids.update(self.make_chunk_id(item.chunk) for item in chunks)

    def count(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        self.clear_called = True
        self.ids.clear()


class FakeEmbeddingService:
    def __init__(
        self,
        outcomes: list[Exception | None] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [])
        self.events = events
        self.calls: list[list[MarkdownChunk]] = []

    def embed_chunks(self, chunks: list[MarkdownChunk]) -> list[EmbeddedChunk]:
        chunk_list = list(chunks)
        self.calls.append(chunk_list)
        if self.events is not None:
            self.events.append("embed")

        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if outcome is not None:
                raise outcome

        return [
            EmbeddedChunk(chunk=chunk, embedding=[1.0, 0.0, 0.0])
            for chunk in chunk_list
        ]


def run_indexing(
    chunks: list[MarkdownChunk],
    store: FakeStore,
    service: FakeEmbeddingService,
    **kwargs: object,
):
    return index_chunks(
        chunks,
        store,  # type: ignore[arg-type]
        embedding_service_factory=lambda: service,  # type: ignore[arg-type]
        batch_delay=0,
        retry_delay=0,
        output=Mock(),
        **kwargs,
    )


def test_existing_chunks_are_ignored_before_embedding() -> None:
    chunks = make_chunks(3)
    store = FakeStore({"docs/document.md::0"})
    service = FakeEmbeddingService()

    stats = run_indexing(chunks, store, service, batch_size=10)

    assert stats.previously_indexed == 1
    assert [chunk.chunk_index for chunk in service.calls[0]] == [1, 2]


def test_only_missing_chunks_receive_embeddings() -> None:
    chunks = make_chunks(4)
    store = FakeStore({"docs/document.md::0", "docs/document.md::2"})
    service = FakeEmbeddingService()

    run_indexing(chunks, store, service, batch_size=10)

    embedded_indexes = [chunk.chunk_index for chunk in service.calls[0]]
    assert embedded_indexes == [1, 3]
    assert store.existing_ids_calls == 1


def test_partially_filled_index_resumes_pending_chunks() -> None:
    chunks = make_chunks(5)
    store = FakeStore({"docs/document.md::0", "docs/document.md::1"})
    service = FakeEmbeddingService()

    stats = run_indexing(chunks, store, service, batch_size=2)

    assert stats.new_chunks_indexed == 3
    assert stats.stored_chunks == 5
    assert stats.remaining_chunks == 0


def test_batches_respect_configured_size() -> None:
    service = FakeEmbeddingService()

    run_indexing(make_chunks(5), FakeStore(), service, batch_size=2)

    assert [len(batch) for batch in service.calls] == [2, 2, 1]


def test_each_batch_is_persisted_before_next_embedding() -> None:
    events: list[str] = []
    store = FakeStore(events=events)
    service = FakeEmbeddingService(events=events)

    run_indexing(make_chunks(3), store, service, batch_size=2)

    assert events == ["embed", "persist", "embed", "persist"]


def test_later_failure_keeps_previous_batches_persisted() -> None:
    store = FakeStore()
    service = FakeEmbeddingService([None, EmbeddingError("unexpected failure")])

    with pytest.raises(EmbeddingError, match="unexpected failure"):
        run_indexing(make_chunks(4), store, service, batch_size=2)

    assert store.ids == {"docs/document.md::0", "docs/document.md::1"}


def test_clear_forces_all_chunks_to_be_reindexed() -> None:
    chunks = make_chunks(3)
    existing = {f"docs/document.md::{index}" for index in range(3)}
    store = FakeStore(existing)
    service = FakeEmbeddingService()

    stats = run_indexing(chunks, store, service, clear=True, batch_size=3)

    assert store.clear_called is True
    assert stats.previously_indexed == 0
    assert len(service.calls[0]) == 3


def test_quota_error_retries_after_configured_delay() -> None:
    service = FakeEmbeddingService(
        [EmbeddingError("429 RESOURCE_EXHAUSTED"), None]
    )
    sleep = Mock()

    stats = index_chunks(
        make_chunks(1),
        FakeStore(),  # type: ignore[arg-type]
        batch_size=1,
        batch_delay=0,
        retry_delay=65,
        max_retries=1,
        embedding_service_factory=lambda: service,  # type: ignore[arg-type]
        sleep=sleep,
        output=Mock(),
    )

    assert stats.completed is True
    assert len(service.calls) == 2
    sleep.assert_called_once_with(65)


def test_quota_retry_respects_maximum() -> None:
    quota_error = EmbeddingError("429 RESOURCE_EXHAUSTED")
    service = FakeEmbeddingService([quota_error, quota_error, quota_error])
    sleep = Mock()

    stats = index_chunks(
        make_chunks(1),
        FakeStore(),  # type: ignore[arg-type]
        batch_size=1,
        batch_delay=0,
        retry_delay=10,
        max_retries=2,
        embedding_service_factory=lambda: service,  # type: ignore[arg-type]
        sleep=sleep,
        output=Mock(),
    )

    assert stats.completed is False
    assert stats.remaining_chunks == 1
    assert len(service.calls) == 3
    assert sleep.call_count == 2


def test_success_after_quota_retry_continues_next_batch() -> None:
    service = FakeEmbeddingService(
        [EmbeddingError("RESOURCE_EXHAUSTED: 429"), None, None]
    )

    stats = run_indexing(
        make_chunks(2),
        FakeStore(),
        service,
        batch_size=1,
        max_retries=1,
    )

    assert stats.completed is True
    assert stats.new_chunks_indexed == 2
    assert len(service.calls) == 3


def test_fully_indexed_collection_does_not_create_embedding_service() -> None:
    chunks = make_chunks(2)
    existing = {"docs/document.md::0", "docs/document.md::1"}
    factory = Mock()

    stats = index_chunks(
        chunks,
        FakeStore(existing),  # type: ignore[arg-type]
        embedding_service_factory=factory,
        output=Mock(),
    )

    assert stats.pending_chunks == 0
    assert stats.new_chunks_indexed == 0
    factory.assert_not_called()


def test_cli_parses_batch_size_and_delay_overrides() -> None:
    args = parse_args(["--batch-size", "25", "--delay", "40", "--clear"])

    assert args.batch_size == 25
    assert args.delay == 40
    assert args.clear is True
