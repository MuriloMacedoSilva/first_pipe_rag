from functools import lru_cache

from fastapi import HTTPException, status

from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService
from app.rag.llm import GoogleLLMService, LLMError
from app.rag.rag_service import RAGService
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore


@lru_cache(maxsize=1)
def _build_rag_service() -> RAGService:
    embedding_service = GoogleEmbeddingService()
    vector_store = ChromaVectorStore()
    retriever = RAGRetriever(embedding_service, vector_store)
    llm_service = GoogleLLMService()
    return RAGService(retriever, llm_service)


def get_rag_service() -> RAGService:
    try:
        return _build_rag_service()
    except (EmbeddingError, LLMError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from None
