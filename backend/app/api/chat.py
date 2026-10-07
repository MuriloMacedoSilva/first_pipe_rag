from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import ChatRequest, ChatResponse, SourceResponse
from app.config import RAG_TOP_K
from app.dependencies import get_rag_service
from app.rag.embeddings import EmbeddingError
from app.rag.llm import LLMError
from app.rag.rag_service import RAGService


router = APIRouter(tags=["chat"])


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "resource_exhausted" in message


def _is_configuration_error(error: Exception) -> bool:
    message = str(error).lower()
    return "api key is required" in message or "model is required" in message


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question using the indexed Markdown material",
)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    top_k = request.top_k if request.top_k is not None else RAG_TOP_K

    try:
        response = rag_service.ask(request.message, top_k=top_k)
        sources = [
            SourceResponse(
                filename=source.filename,
                relative_path=source.relative_path,
                section=source.section,
                chunk_index=source.chunk_index,
                distance=source.distance,
            )
            for source in response.sources
        ]
        return ChatResponse(answer=response.answer, sources=sources)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat request.",
        ) from None
    except (EmbeddingError, LLMError) as error:
        if _is_quota_error(error):
            detail = "AI service quota is temporarily unavailable."
        elif _is_configuration_error(error):
            detail = "AI service is not configured."
        else:
            detail = "AI service is temporarily unavailable."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from None
