"""Vectors for hybrid search: dense (sentence-transformers) and sparse (BM25).

Dense embeddings come from ``embedding.dense_model`` in ``configs/ingestion.yaml``
(``BAAI/bge-base-en-v1.5``: 768 dimensions, normalized so a dot product is the cosine
similarity, which is what the ``dotproduct`` Pinecone index expects).

BGE models are trained to see an instruction in front of *queries* -- not documents -- for
retrieval, so :meth:`DenseEmbedder.embed_query` adds it and
:meth:`DenseEmbedder.embed_documents` does not.

``sentence-transformers`` (and PyTorch) are only imported when a model is first loaded.

Sparse vectors are BM25 term weights (:class:`BM25Encoder`), fit on the chunk corpus and
saved to ``embedding.sparse_params_path`` (``data/processed/bm25.json``) so queries are
weighted with the same statistics as the indexed chunks. They catch what dense vectors
blur: exact invoice numbers, PO references, amounts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.config import EmbeddingConfig, get_config, repo_path
from ledgerlens.ingestion.chunking import Chunk

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


# --------------------------------------------------------------------------- sparse (BM25)

# Lower-cased words and numbers; joiners inside a token are kept, so "INV-2026-0417",
# "498.00", "1,234.50" and "14/03/2026" each stay one searchable token.
_BM25_TOKEN = re.compile(r"[a-z0-9]+(?:[.,/\-][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "your",
        "our",
        "we",
        "you",
    ]
)
BM25_PARAMS_VERSION = 1


def bm25_tokens(text: str) -> list[str]:
    """BM25 terms of ``text``: lower-cased, stopwords and single letters dropped."""
    return [
        token
        for token in _BM25_TOKEN.findall(text.lower())
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def token_index(token: str) -> int:
    """Stable 32-bit index for a term (the sparse vector dimension it lands in)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


class SparseVector(BaseModel):
    """Pinecone's sparse format: parallel lists of dimensions and weights."""

    model_config = ConfigDict(frozen=True)

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.indices)


class BM25Encoder:
    """Okapi BM25 term weights as sparse vectors, fit on the chunk corpus.

    Documents get saturated, length-normalized term frequencies; queries get each term's
    inverse document frequency (normalized to sum to 1), so the dot product of the two is
    the BM25 score. Same scheme as ``pinecone-text``'s ``BM25Encoder``, but with a
    tokenizer that keeps document numbers whole and needs no NLTK downloads.
    """

    def __init__(self, *, b: float = 0.75, k1: float = 1.2) -> None:
        self.b = b
        self.k1 = k1
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_freq: dict[int, int] = {}

    @property
    def is_fitted(self) -> bool:
        return self.n_docs > 0

    def fit(self, corpus: Iterable[str]) -> BM25Encoder:
        """Learn document frequencies and average length from ``corpus`` (replacing any)."""
        doc_freq: Counter[int] = Counter()
        n_docs = total_len = 0
        for text in corpus:
            tokens = bm25_tokens(text)
            n_docs += 1
            total_len += len(tokens)
            doc_freq.update({token_index(t) for t in tokens})
        if n_docs == 0:
            raise ValueError("cannot fit BM25 on an empty corpus")
        self.n_docs, self.avgdl, self.doc_freq = n_docs, total_len / n_docs, dict(doc_freq)
        return self

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        self._require_fit()
        return [self._encode_document(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        self._require_fit()
        terms = sorted({token_index(t) for t in bm25_tokens(text)})
        idf = [math.log((self.n_docs + 1) / (self.doc_freq.get(i, 0) + 0.5)) for i in terms]
        total = sum(idf)
        if total <= 0:
            return SparseVector()
        return SparseVector(indices=terms, values=[w / total for w in idf])

    def _encode_document(self, text: str) -> SparseVector:
        counts = Counter(token_index(t) for t in bm25_tokens(text))
        length = sum(counts.values())
        norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
        indices = sorted(counts)
        values = [counts[i] * (self.k1 + 1) / (counts[i] + norm) for i in indices]
        return SparseVector(indices=indices, values=values)

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("BM25Encoder is not fitted: call fit() or load() first")

    # ------------------------------------------------------------------ persistence

    def to_params(self) -> dict[str, Any]:
        indices = sorted(self.doc_freq)
        return {
            "version": BM25_PARAMS_VERSION,
            "tokenizer": "ledgerlens-regex",
            "hash": "blake2b-32",
            "b": self.b,
            "k1": self.k1,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_freq": {"indices": indices, "values": [self.doc_freq[i] for i in indices]},
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> BM25Encoder:
        if params.get("version") != BM25_PARAMS_VERSION:
            raise ValueError(f"unsupported BM25 params version {params.get('version')!r}")
        encoder = cls(b=float(params["b"]), k1=float(params["k1"]))
        encoder.n_docs = int(params["n_docs"])
        encoder.avgdl = float(params["avgdl"])
        df = params["doc_freq"]
        encoder.doc_freq = {
            int(i): int(v) for i, v in zip(df["indices"], df["values"], strict=True)
        }
        return encoder

    def save(self, path: Path | str | None = None) -> Path:
        """Write the fitted params (default: ``embedding.sparse_params_path``)."""
        self._require_fit()
        target = _params_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_params()), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> BM25Encoder:
        """Read params written by :meth:`save` (default: ``embedding.sparse_params_path``)."""
        return cls.from_params(json.loads(_params_path(path).read_text(encoding="utf-8")))


def _params_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    return repo_path(get_config().ingestion.embedding.sparse_params_path)


# --------------------------------------------------------------------------- chunks -> vectors

MetadataValue = str | int | float | bool | list[str]


class EmbeddedChunk(BaseModel):
    """One chunk ready for the vector store: dense + sparse vectors and filterable metadata."""

    model_config = ConfigDict(frozen=True)

    id: str  # "{document_id}#c{index}"
    values: Vector
    sparse: SparseVector
    metadata: dict[str, MetadataValue]

    def to_pinecone(self) -> dict[str, Any]:
        """The record shape Pinecone's upsert takes (sparse part omitted when empty)."""
        record: dict[str, Any] = {"id": self.id, "values": self.values, "metadata": self.metadata}
        if self.sparse.indices:
            record["sparse_values"] = {
                "indices": self.sparse.indices,
                "values": self.sparse.values,
            }
        return record


def chunk_metadata(
    document_id: str, chunk: Chunk, extra: Mapping[str, MetadataValue] | None = None
) -> dict[str, MetadataValue]:
    """Metadata stored with a chunk's vector. Pinecone rejects nulls, so unset fields are
    left out, and list values must be strings."""
    metadata: dict[str, MetadataValue] = {
        "document_id": document_id,
        "chunk_index": chunk.index,
        "doc_type": chunk.doc_type.value,
        "kind": chunk.kind,
        "pages": [str(page) for page in chunk.pages],
        "text": chunk.text,
    }
    if chunk.title:
        metadata["title"] = chunk.title
    if extra:
        metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata


class ChunkEmbedder:
    """Dense + sparse vectors for a document's chunks in one call."""

    def __init__(self, dense: DenseEmbedder, sparse: BM25Encoder) -> None:
        self.dense = dense
        self.sparse = sparse

    def embed(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> list[EmbeddedChunk]:
        """Embed ``chunks`` of ``document_id``; ``metadata`` is added to every chunk's."""
        texts = [chunk.text for chunk in chunks]
        dense = self.dense.embed_documents(texts)
        sparse = self.sparse.encode_documents(texts)
        return [
            EmbeddedChunk(
                id=chunk.vector_id(document_id),
                values=values,
                sparse=sparse_vector,
                metadata=chunk_metadata(document_id, chunk, metadata),
            )
            for chunk, values, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
