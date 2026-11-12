"""Vectors for hybrid search: dense (sentence-transformers) and sparse (BM25).

Dense embeddings come from ``embedding.dense_model`` in ``configs/ingestion.yaml``
(``BAAI/bge-base-en-v1.5``: 768 dimensions, normalized so a dot product is the cosine
similarity, which is what the ``dotproduct`` Pinecone index expects).

BGE models are trained to see an instruction in front of *queries* -- not documents -- for
retrieval, so :meth:`DenseEmbedder.embed_query` adds it and
:meth:`DenseEmbedder.embed_documents` does not.

``sentence-transformers`` (and PyTorch) are only imported when a model is first loaded.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ledgerlens.core.config import EmbeddingConfig, get_config

Vector = list[float]

# https://huggingface.co/BAAI/bge-base-en-v1.5#using-sentence-transformers
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceEncoder(Protocol):
    """The part of ``sentence_transformers.SentenceTransformer`` the embedder uses."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any: ...


def query_instruction_for(model_name: str) -> str:
    """The query prefix a model expects ('' for models that do not use one)."""
    name = model_name.lower()
    if "bge" in name and "-en" in name and "m3" not in name:
        return BGE_QUERY_INSTRUCTION
    return ""


def load_sentence_transformer(model_name: str, device: str | None = None) -> SentenceEncoder:
    """Load ``model_name`` with sentence-transformers (downloads it the first time)."""
    module = importlib.import_module("sentence_transformers")
    model: SentenceEncoder = module.SentenceTransformer(model_name, device=device)
    return model


class DenseEmbedder:
    """Dense vectors for chunks and queries.

    Pass ``model`` to use an already-loaded (or fake) encoder; otherwise
    ``embedding.dense_model`` is loaded on first use.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: SentenceEncoder | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        self.config = config or get_config().ingestion.embedding
        self.device = device
        self._model = model
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else query_instruction_for(self.config.dense_model)
        )

    @property
    def model(self) -> SentenceEncoder:
        if self._model is None:
            self._model = load_sentence_transformer(self.config.dense_model, self.device)
        return self._model

    @property
    def dimension(self) -> int:
        # sentence-transformers 6 renamed get_sentence_embedding_dimension (now deprecated).
        getter = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        dim = getter() if callable(getter) else None
        if not isinstance(dim, int):
            raise ValueError(f"{self.config.dense_model} does not report its dimension")
        return dim

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, in order (no query instruction)."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Vector:
        """The vector for a search query, with the model's query instruction prepended."""
        return self._encode([self.query_instruction + text])[0]

    def _encode(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        array: npt.NDArray[np.float32] = np.asarray(
            self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        vectors: list[Vector] = array.tolist()
        return vectors
