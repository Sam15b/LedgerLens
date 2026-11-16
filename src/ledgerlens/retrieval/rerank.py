"""Cross-encoder reranking: score each (question, chunk) pair jointly, then re-sort.

A bi-encoder (the dense model) embeds the question and the chunk separately, so it can
only compare two points in vector space; a cross-encoder reads both together and sees,
for example, that the question asks for ``INV-889`` and this chunk *is* ``INV-889``. It is
far too slow to run over the index, so R3 runs it on R2's top 20 only.

Model: ``BAAI/bge-reranker-base`` (``DEFAULT_RERANK_MODEL``), loaded with
sentence-transformers on first use and shared by every :class:`CrossEncoderReranker` in
the process. sentence-transformers applies the model's activation (a sigmoid for this
one-label model), so scores read as relevance in (0, 1).
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Sequence
from typing import Any, Protocol

from ledgerlens.retrieval import ScoredChunk

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class PairScorer(Protocol):
    """The part of ``sentence_transformers.CrossEncoder`` the reranker uses."""

    def predict(self, inputs: list[tuple[str, str]], /, **kwargs: Any) -> Any: ...


@functools.lru_cache(maxsize=2)
def load_cross_encoder(
    model_name: str, device: str | None = None, max_length: int = 512
) -> PairScorer:
    """Load a cross-encoder with sentence-transformers (downloaded the first time)."""
    module = importlib.import_module("sentence_transformers")
    model: PairScorer = module.CrossEncoder(model_name, device=device, max_length=max_length)
    return model


class CrossEncoderReranker:
    """Re-sorts chunks by a cross-encoder's relevance to the question."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        model: PairScorer | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model

    @property
    def model(self) -> PairScorer:
        if self._model is None:
            self._model = load_cross_encoder(self.model_name, self.device, self.max_length)
        return self._model

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to ``query`` (higher is more relevant)."""
        if not texts:
            return []
        raw = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in raw]

    def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        """``chunks`` best first by cross-encoder score (ties keep their earlier order),
        cut to ``top_n``. Each chunk's ``score`` becomes its rerank score; the score it came
        with stays in ``scores``."""
        relevance = self.scores(query, [chunk.text for chunk in chunks])
        rescored = [
            chunk.model_copy(update={"score": score, "scores": chunk.scores | {"rerank": score}})
            for chunk, score in zip(chunks, relevance, strict=True)
        ]
        order = sorted(range(len(rescored)), key=lambda i: (-rescored[i].score, i))
        ranked = [rescored[i] for i in order]
        return ranked if top_n is None else ranked[:top_n]
