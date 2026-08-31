"""FAISS-backed semantic search over the phone document corpus.

Embeddings come from `sentence-transformers/all-MiniLM-L6-v2` — small, open
source, and strong enough for the short factual passages in `documents.py`.

Vectors are L2-normalised and stored in an inner-product index, which makes the
returned score exact cosine similarity. That matters downstream: the chatbot
treats a low top-score as "the corpus does not cover this" and says so instead
of answering from a weak match.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

import config

from .documents import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """Embeds documents and retrieves the nearest ones for a query."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self._index = None
        self.documents: list[Document] = []

    # ------------------------------------------------------------------
    @property
    def model(self):
        """Load the embedding model lazily so imports stay cheap."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._index is not None and bool(self.documents)

    # ------------------------------------------------------------------
    def build(self, documents: list[Document]) -> None:
        """Embed the corpus and construct the FAISS index."""
        import faiss

        if not documents:
            raise ValueError("Cannot build an index from an empty corpus")

        self.documents = documents
        texts = [doc.text for doc in documents]

        logger.info("Embedding %s documents", len(texts))
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Inner product over unit vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        logger.info("Index built: %s vectors, dim %s", self._index.ntotal, vectors.shape[1])

    def search(
        self, query: str, top_k: int | None = None, phone_ids: list[int] | None = None
    ) -> list[tuple[Document, float]]:
        """Return `(document, cosine_score)` pairs, best first.

        `phone_ids` restricts results to specific handsets. The chatbot uses it
        once it has resolved which models a question is about, so a query like
        "camera specs" cannot drift onto a phone the user never mentioned.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty — call build() or load() first")

        top_k = top_k or config.RAG_TOP_K

        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # Over-fetch before filtering, otherwise the filter can empty the result.
        fetch = min(len(self.documents), top_k * 6 if phone_ids else top_k)
        scores, indices = self._index.search(query_vector, fetch)

        allowed = set(phone_ids) if phone_ids else None
        results: list[tuple[Document, float]] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self.documents[index]
            if allowed is not None and document.phone_id not in allowed:
                continue
            results.append((document, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None, fingerprint: str | None = None) -> None:
        """Persist index and documents so restarts skip re-embedding."""
        import faiss

        path = Path(path or config.VECTOR_INDEX_PATH)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "documents.pkl", "wb") as handle:
            pickle.dump(
                {
                    "documents": self.documents,
                    "model_name": self.model_name,
                    "fingerprint": fingerprint,
                },
                handle,
            )
        logger.info("Vector store saved to %s", path)

    def load(self, path: Path | None = None, fingerprint: str | None = None) -> bool:
        """Restore a saved index. Returns False when there is nothing usable.

        `fingerprint` identifies the database contents the index was built from.
        A mismatch means the index is stale — after `scrape.py --reset`, for
        instance, phone rows are reassigned new ids while the cached documents
        still carry the old ones. Retrieval would then filter on ids that point
        at different phones and quietly answer about the wrong handset, so a
        changed fingerprint forces a rebuild.
        """
        import faiss

        path = Path(path or config.VECTOR_INDEX_PATH)
        index_file = path / "index.faiss"
        docs_file = path / "documents.pkl"

        if not index_file.exists() or not docs_file.exists():
            return False

        try:
            with open(docs_file, "rb") as handle:
                payload = pickle.load(handle)

            # A cached index built with different embeddings would return
            # meaningless neighbours, so treat it as absent.
            if payload.get("model_name") != self.model_name:
                logger.warning(
                    "Cached index used %s but config asks for %s — rebuilding",
                    payload.get("model_name"),
                    self.model_name,
                )
                return False

            if fingerprint is not None and payload.get("fingerprint") != fingerprint:
                logger.warning("Database changed since the index was built — rebuilding")
                return False

            self.documents = payload["documents"]
            self._index = faiss.read_index(str(index_file))
            logger.info("Loaded %s documents from %s", len(self.documents), path)
            return True
        except Exception as exc:
            logger.warning("Could not load vector store (%s) — rebuilding", exc)
            return False

    def stats(self) -> dict:
        return {
            "ready": self.is_ready,
            "documents": len(self.documents),
            "vectors": int(self._index.ntotal) if self._index is not None else 0,
            "embedding_model": self.model_name,
        }
