import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    EMBEDDING_BATCH_DELAY_SECONDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY_SECONDS,
)
from app.rag.embeddings import EmbeddedChunk, EmbeddingError, GoogleEmbeddingService
from app.rag.markdown_chunker import MarkdownChunk, MarkdownChunker
from app.rag.markdown_loader import MarkdownLoader
from app.rag.vector_store import ChromaVectorStore


@dataclass
class IndexingStats:
    total_chunks: int
    previously_indexed: int
    pending_chunks: int
    new_chunks_indexed: int
    stored_chunks: int
    remaining_chunks: int
    completed: bool


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally index Markdown material with Google and ChromaDB."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the collection before indexing all material.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the number of chunks embedded per batch.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        help="Override the delay in seconds between successful batches.",
    )
    return parser.parse_args(args)


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "resource_exhausted" in message


def index_chunks(
    chunks: Sequence[MarkdownChunk],
    store: ChromaVectorStore,
    *,
    clear: bool = False,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    batch_delay: float = EMBEDDING_BATCH_DELAY_SECONDS,
    retry_delay: float = EMBEDDING_RETRY_DELAY_SECONDS,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    embedding_service_factory: Callable[[], GoogleEmbeddingService] = (
        GoogleEmbeddingService
    ),
    sleep: Callable[[float], None] = time.sleep,
    output: Callable[[str], None] = print,
) -> IndexingStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if batch_delay < 0 or retry_delay < 0:
        raise ValueError("batch delays cannot be negative")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    if clear:
        store.clear()

    chunk_ids = [store.make_chunk_id(chunk) for chunk in chunks]
    existing_ids = store.existing_ids(chunk_ids)
    pending = [
        chunk
        for chunk, chunk_id in zip(chunks, chunk_ids, strict=True)
        if chunk_id not in existing_ids
    ]
    previously_indexed = len(chunks) - len(pending)

    output(f"Chunks generated: {len(chunks)}")
    output(f"Already indexed: {previously_indexed}")
    output(f"Pending chunks: {len(pending)}")

    if not pending:
        output("Nothing to index.")
        return IndexingStats(
            total_chunks=len(chunks),
            previously_indexed=previously_indexed,
            pending_chunks=0,
            new_chunks_indexed=0,
            stored_chunks=store.count(),
            remaining_chunks=0,
            completed=True,
        )

    service = embedding_service_factory()
    total_batches = math.ceil(len(pending) / batch_size)
    new_chunks_indexed = 0

    for batch_number, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]
        output("")
        output(f"Batch {batch_number}/{total_batches}")
        output(f"Embedding {len(batch)} chunks...")
        retries = 0

        while True:
            try:
                embedded_chunks = service.embed_chunks(batch)
                break
            except EmbeddingError as error:
                if not is_quota_error(error):
                    raise
                if retries >= max_retries:
                    stored_chunks = store.count()
                    remaining = len(pending) - new_chunks_indexed
                    output("")
                    output("Indexing interrupted due to API quota.")
                    output(f"Total chunks: {len(chunks)}")
                    output(f"Stored: {stored_chunks}")
                    output(f"Remaining: {remaining}")
                    output("Run the same command again to resume.")
                    return IndexingStats(
                        total_chunks=len(chunks),
                        previously_indexed=previously_indexed,
                        pending_chunks=len(pending),
                        new_chunks_indexed=new_chunks_indexed,
                        stored_chunks=stored_chunks,
                        remaining_chunks=remaining,
                        completed=False,
                    )

                retries += 1
                output(
                    "API quota reached. "
                    f"Retrying in {retry_delay:g} seconds "
                    f"({retries}/{max_retries})..."
                )
                sleep(retry_delay)

        store.add_chunks(embedded_chunks)
        new_chunks_indexed += len(embedded_chunks)
        output(f"Stored {len(embedded_chunks)} chunks.")
        output(
            f"Progress: {previously_indexed + new_chunks_indexed}/{len(chunks)}"
        )

        if batch_number < total_batches and batch_delay > 0:
            output("Waiting before next batch...")
            sleep(batch_delay)

    return IndexingStats(
        total_chunks=len(chunks),
        previously_indexed=previously_indexed,
        pending_chunks=len(pending),
        new_chunks_indexed=new_chunks_indexed,
        stored_chunks=store.count(),
        remaining_chunks=0,
        completed=True,
    )


def main() -> None:
    args = parse_args()
    batch_size = (
        args.batch_size if args.batch_size is not None else EMBEDDING_BATCH_SIZE
    )
    batch_delay = (
        args.delay if args.delay is not None else EMBEDDING_BATCH_DELAY_SECONDS
    )
    project_root = Path(__file__).resolve().parents[2]
    material_path = project_root / "material"

    documents = MarkdownLoader(material_path).load()
    chunks = MarkdownChunker().chunk_documents(documents)
    store = ChromaVectorStore()

    print(f"Documents loaded: {len(documents)}")
    stats = index_chunks(
        chunks,
        store,
        clear=args.clear,
        batch_size=batch_size,
        batch_delay=batch_delay,
    )

    if not stats.completed:
        return

    print("")
    print("Indexing completed.")
    print(f"Documents: {len(documents)}")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Previously indexed: {stats.previously_indexed}")
    print(f"New chunks indexed: {stats.new_chunks_indexed}")
    print(f"Chunks stored: {stats.stored_chunks}")
    print(f"Collection: {store.collection_name}")


if __name__ == "__main__":
    main()
