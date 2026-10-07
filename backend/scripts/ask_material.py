import argparse
from collections.abc import Sequence

from app.config import RAG_TOP_K
from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService
from app.rag.llm import GoogleLLMService, LLMError
from app.rag.rag_service import RAGService
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer questions using only the indexed Markdown material."
    )
    parser.add_argument("question", help="Question to answer from the documents.")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Maximum number of retrieved chunks. Overrides RAG_TOP_K.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieval distances without exposing prompts or embeddings.",
    )
    return parser.parse_args(args)


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "resource_exhausted" in message


def main() -> None:
    args = parse_args()
    top_k = args.top_k if args.top_k is not None else RAG_TOP_K

    try:
        vector_store = ChromaVectorStore()
        embedding_service = GoogleEmbeddingService()
        retriever = RAGRetriever(embedding_service, vector_store)
        llm_service = GoogleLLMService()
        rag_service = RAGService(retriever, llm_service)
        response = rag_service.ask(args.question, top_k=top_k)
    except (EmbeddingError, LLMError) as error:
        if is_quota_error(error):
            raise SystemExit(
                "Google API quota exceeded. Try again after the quota resets."
            ) from None
        raise SystemExit(f"RAG request failed: {error}") from None
    except ValueError as error:
        raise SystemExit(f"Invalid request: {error}") from None

    print("Question:")
    print(args.question)
    print("")
    print("Answer:")
    print(response.answer)
    print("")
    print("Sources:")

    if not response.sources:
        print("No sources found.")

    for position, source in enumerate(response.sources, start=1):
        print("")
        print(f"{position}. {source.relative_path}")
        print(f"   Section: {source.section or '(none)'}")
        print(f"   Chunk: {source.chunk_index}")

    if args.debug:
        print("")
        print(f"Debug: {len(response.sources)} chunks retrieved")
        for position, source in enumerate(response.sources, start=1):
            distance = (
                f"{source.distance:.4f}" if source.distance is not None else "N/A"
            )
            print(f"{position}. Distance: {distance}")


if __name__ == "__main__":
    main()
