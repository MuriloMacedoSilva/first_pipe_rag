from app.rag.embeddings import EmbeddingError, GoogleEmbeddingService


def main() -> None:
    try:
        embedding = GoogleEmbeddingService().embed_text(
            "Teste de embedding do projeto RAG."
        )
    except (EmbeddingError, ValueError) as error:
        raise SystemExit(f"Embedding test failed: {error}") from None

    print("Embedding generated successfully")
    print(f"Dimensions: {len(embedding)}")
    print(f"First values: {[round(value, 6) for value in embedding[:5]]}")


if __name__ == "__main__":
    main()
