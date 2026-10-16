import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Empty means "pick the first available model" (see llm_provider.get_default_model).
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or None
HOST = os.getenv("HOST","127.0.0.1")
PORT = os.getenv("PORT","8000")

MAX_MEMORY_MESSAGES = os.getenv("MAX_MEMORY_MESSAGES","20")
RAG_TOP_K = os.getenv("RAG_TOP_K","5")
RAG_CHUNK_SIZE = os.getenv("RAG_CHUNK_SIZE","800")
RAG_CHUNK_OVERLAP = os.getenv("RAG_CHUNK_OVERLAP","100")
# "tfidf" (local, free) or "openai" (text-embedding-3-small, needs paid OpenAI credits).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf").strip().lower()
# Chunks scoring below this are treated as unrelated and not sent to the model.
# Empty = backend default (0.05 for tfidf, 0.30 for openai embeddings).
RAG_MIN_SCORE = os.getenv("RAG_MIN_SCORE") or None
# Gemini 3.x "thinking" depth: minimal | low | medium | high. Lower = faster replies.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().upper()
# How many other models to try if the chosen one fails (0 = no fallback).
MAX_FALLBACKS = os.getenv("MAX_FALLBACKS", "2")

DATA_DIR = os.getenv("DATA_DIR","data")