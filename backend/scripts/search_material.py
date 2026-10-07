import argparse
from collections.abc import Sequence

from app.config import RAG_TOP_K
from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


PREVIEW_LENGTH = 500
SEPARATOR = "-" * 50


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed Markdown material by semantic similarity."
    )
    parser.add_argument("query", help="Question or text to search for.")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Maximum number of results. Overrides RAG_TOP_K.",
    )
    return parser.parse_args(args)


def content_preview(content: str) -> str:
    content = content.strip()
    if len(content) <= PREVIEW_LENGTH:
        return content
    return f"{content[:PREVIEW_LENGTH]}..."


def print_result(position: int, result: VectorSearchResult) -> None:
    distance = f"{result.distance:.4f}" if result.distance is not None else "N/A"
    print(f"[{position}]")
    print(f"File: {result.relative_path}")
    print(f"Section: {result.section or '(none)'}")
    print(f"Chunk index: {result.chunk_index}")
    print(f"Distance: {distance}")
    print("")
    print("Preview:")
    print(content_preview(result.content))
    print("")
    print(SEPARATOR)


def main() -> None:
    args = parse_args()
    top_k = args.top_k if args.top_k is not None else RAG_TOP_K

    vector_store = ChromaVectorStore()
    embedding_service = GoogleEmbeddingService()
    retriever = RAGRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    try:
        results = retriever.retrieve(args.query, top_k=top_k)
    except (EmbeddingError, ValueError) as error:
        raise SystemExit(f"Search failed: {error}") from None

    print("Query:")
    print(args.query)
    print("")
    print(f"Results: {len(results)}")
    print("")

    for position, result in enumerate(results, start=1):
        print_result(position, result)


if __name__ == "__main__":
    main()
