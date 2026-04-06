"""Local embeddings — fastembed + numpy cosine similarity.

Stores .npy vectors alongside memory files. No external services needed.
Model downloads ~100MB on first use, then cached locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGS_SUBDIR = ".embeddings"
INDEX_FILE = "_index.json"


class EmbeddingStore:
    """File-backed vector store using fastembed for local embedding generation."""

    def __init__(self, memory_dir: Path, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.memory_dir = memory_dir
        self.embed_dir = memory_dir / EMBEDDINGS_SUBDIR
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._index = self._load_index()

    def _get_model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _load_index(self) -> dict[str, str]:
        """Load the path -> npy filename index."""
        idx_path = self.embed_dir / INDEX_FILE
        if idx_path.exists():
            return json.loads(idx_path.read_text())
        return {}

    def _save_index(self) -> None:
        (self.embed_dir / INDEX_FILE).write_text(json.dumps(self._index))

    def _safe_filename(self, path: str) -> str:
        return path.replace("/", "__").replace(" ", "_").replace(".md", "") + ".npy"

    def embed_and_store(self, path: str, content: str) -> None:
        """Generate embedding for content and store as .npy file."""
        if not content.strip():
            return
        model = self._get_model()
        embeddings = list(model.embed([content]))
        vec = np.array(embeddings[0], dtype=np.float32)

        fname = self._safe_filename(path)
        np.save(self.embed_dir / fname, vec)
        self._index[path] = fname
        self._save_index()

    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[tuple[str, float]]:
        """Find memory paths most similar to query by cosine similarity."""
        if not self._index:
            return []

        model = self._get_model()
        q_vec = np.array(list(model.embed([query]))[0], dtype=np.float32)

        scores: list[tuple[str, float]] = []
        for mem_path, fname in self._index.items():
            npy_path = self.embed_dir / fname
            if not npy_path.exists():
                continue
            vec = np.load(npy_path)
            sim = float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-8))
            if sim >= threshold:
                scores.append((mem_path, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete(self, path: str) -> None:
        """Remove embedding for a memory path."""
        fname = self._index.pop(path, None)
        if fname:
            npy_path = self.embed_dir / fname
            if npy_path.exists():
                npy_path.unlink()
            self._save_index()
