import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
)
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.5-flash-lite")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "markdown_docs")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
EMBEDDING_BATCH_DELAY_SECONDS = float(
    os.getenv("EMBEDDING_BATCH_DELAY_SECONDS", "35")
)
EMBEDDING_RETRY_DELAY_SECONDS = float(
    os.getenv("EMBEDDING_RETRY_DELAY_SECONDS", "65")
)
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.85"))
RAG_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "3"))
RAG_MIN_CONTENT_CHARS = int(os.getenv("RAG_MIN_CONTENT_CHARS", "40"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
