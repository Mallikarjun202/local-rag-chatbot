
import os
from typing import Any

import numpy as np
from . import config
import pickle
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunk_size = max(1, chunk_size)
    # Overlap must be smaller than the chunk, or the window never moves forward.
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    while True:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

class EmbeddingBackend:
    name = "base"

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "EmbeddingBackend subclasses must implement the embed method")

class OpenAIEmbedding(EmbeddingBackend):
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model="text-embedding-3-small", input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class TfidfEmbedding(EmbeddingBackend):
    name = "tfidf-local"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
        # Drop "what/is/the"-style words so off-topic questions score ~0.
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed(self, texts: list[str]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        if not self.fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return np.array(matrix.toarray(), dtype=np.float32)


def get_backend() -> EmbeddingBackend:
    # OpenAI embeddings are opt-in: having a key doesn't mean it has credits.
    if config.EMBEDDING_BACKEND == "openai" and config.OPENAI_API_KEY:
        return OpenAIEmbedding()
    return TfidfEmbedding()


class RagCollection:
    def __init__(self, name: str):
        self.name = name
        self.chunks = []
        self.sources = []
        self.embeddings = None
        self.backend = get_backend()
        self._path = os.path.join(config.DATA_DIR, f"{name}.pkl")
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path, "rb") as f:  # context manager for reading the pickle file
                data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self.sources = data.get("sources", [])
            self.embeddings = data.get("embeddings", None)
            saved_backend = data.get("backend")
            # Re-embed when using local TF-IDF (it must be refit) or when the
            # saved vectors came from a different backend (dimensions won't match).
            if self.chunks and (isinstance(self.backend, TfidfEmbedding)
                                or saved_backend != self.backend.name):
                self._reembed_all()

    def _reembed_all(self):
        if isinstance(self.backend, TfidfEmbedding):
            self.backend.fit(self.chunks)
        self.embeddings = self.backend.embed(self.chunks)

    def _save(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(self._path, "wb") as f:  # context manager for writing the pickle file
            pickle.dump({
                "chunks": self.chunks,
                "sources": self.sources,
                "embeddings": self.embeddings,
                "backend": self.backend.name,
            }, f)

    def add_document(self, text: str, source: str = None):
        new_chunks = chunk_text(text, chunk_size=int(
            config.RAG_CHUNK_SIZE), overlap=int(config.RAG_CHUNK_OVERLAP))
        if not new_chunks:
            return 0
        self.chunks.extend(new_chunks)
        self.sources.extend([source] * len(new_chunks))
        if isinstance(self.backend, TfidfEmbedding):
            # Refit on ALL chunks so every chunk shares one vocabulary and
            # embeddings stay aligned with self.chunks.
            self._reembed_all()
        else:
            new_embeddings = self.backend.embed(new_chunks)
            self.embeddings = new_embeddings if self.embeddings is None else np.vstack(
                [self.embeddings, new_embeddings])
        self._save()
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_embedding = self.backend.embed([query])
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:int(top_k)]
        return [(self.chunks[i], self.sources[i], float(sims[i])) for i in top_idx]

    @property
    def min_score(self) -> float:
        if config.RAG_MIN_SCORE is not None:
            return float(config.RAG_MIN_SCORE)
        return 0.05 if isinstance(self.backend, TfidfEmbedding) else 0.30

    def stats(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_chunks": len(self.chunks),
            "sources": sorted(set(self.sources)),
            "embedding_backend": self.backend.name
        }


class RagRegistry:
    def __init__(self):
        self._collections = {}

    def get(self, name: str) -> RagCollection:
        name = name or "default"
        if name not in self._collections:
            self._collections[name] = RagCollection(name)
        return self._collections[name]

    def list_names(self) -> List[str]:
        names = set(self._collections.keys())
        if os.path.isdir(config.DATA_DIR):
            for name in os.listdir(config.DATA_DIR):
                if name.endswith(".pkl"):
                    names.add(name[:-4])  # remove the ".pkl" extension
        return sorted(names)

rag_registry = RagRegistry()