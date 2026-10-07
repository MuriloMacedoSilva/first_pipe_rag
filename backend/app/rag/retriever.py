from app.rag.embeddings import GoogleEmbeddingService
from app.rag.vector_store import ChromaVectorStore, VectorSearchResult


class RAGRetriever:
    def __init__(
        self,
        embedding_service: GoogleEmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_text(query)
        return self.vector_store.query(
            embedding=query_embedding,
            n_results=top_k,
        )
