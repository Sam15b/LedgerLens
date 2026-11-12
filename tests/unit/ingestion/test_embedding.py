import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ledgerlens.core.config import EmbeddingConfig
from ledgerlens.core.ids import new_document_id
from ledgerlens.core.schemas import DocType, parse_extracted
from ledgerlens.ingestion.chunking import Chunk, chunk_document
from ledgerlens.ingestion.embedding import (
    BGE_QUERY_INSTRUCTION,
    BM25Encoder,
    ChunkEmbedder,
    DenseEmbedder,
    SparseVector,
    bm25_tokens,
    length_sorted_batches,
    load_sentence_transformer,
    query_instruction_for,
    token_index,
)
from ledgerlens.ingestion.layout import PageContent, layout_pages

FIXTURES = Path(__file__).parents[2] / "fixtures" / "documents"
TINY_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"  # 128 dims, ~17 MB


class FakeEncoder:
    """Deterministic unit vectors from a hash of the text; records every batch."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        self.batches.append(list(sentences))
        rows = []
        for text in sentences:
            digest = hashlib.sha256(text.encode()).digest()
            row = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32) + 1
            rows.append(row / np.linalg.norm(row) if kwargs["normalize_embeddings"] else row)
        return np.stack(rows)

    def get_embedding_dimension(self) -> int:
        return self.dim


def fixture_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for name in ("invoice_digital", "purchase_order", "contract", "payment_advice"):
        record = parse_extracted((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        pages = [
            p
            for p in layout_pages(FIXTURES / f"{name}.pdf", "application/pdf")
            if isinstance(p, PageContent)
        ]
        chunks += chunk_document(pages, record.doc_type, record=record)
    return chunks


def dot(a: SparseVector, b: SparseVector) -> float:
    weights = dict(zip(b.indices, b.values, strict=True))
    return sum(v * weights.get(i, 0.0) for i, v in zip(a.indices, a.values, strict=True))


# --------------------------------------------------------------------------- dense


def test_queries_get_the_bge_instruction_and_documents_do_not() -> None:
    fake = FakeEncoder()
    embedder = DenseEmbedder(EmbeddingConfig(dense_model="BAAI/bge-base-en-v1.5"), model=fake)

    embedder.embed_documents(["Invoice INV-1"])
    embedder.embed_query("which invoice?")

    assert fake.batches == [["Invoice INV-1"], [BGE_QUERY_INSTRUCTION + "which invoice?"]]


@pytest.mark.parametrize(
    "model,instruction",
    [
        ("BAAI/bge-base-en-v1.5", BGE_QUERY_INSTRUCTION),
        ("BAAI/bge-small-en-v1.5", BGE_QUERY_INSTRUCTION),
        ("BAAI/bge-m3", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_query_instruction_by_model(model: str, instruction: str) -> None:
    assert query_instruction_for(model) == instruction


def test_vectors_are_normalized_and_in_input_order() -> None:
    fake = FakeEncoder()
    embedder = DenseEmbedder(EmbeddingConfig(batch_size=2), model=fake)
    texts = ["bb", "a", "dddd", "ccc", "eeeee"]

    vectors = embedder.embed_documents(texts)

    assert fake.batches == [["eeeee", "dddd"], ["ccc", "bb"], ["a"]]  # longest first
    assert vectors == [embedder.embed_documents([t])[0] for t in texts]
    assert all(math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-6) for v in vectors)
    assert embedder.dimension == 8
    assert embedder.embed_documents([]) == []


def test_length_sorted_batches() -> None:
    assert length_sorted_batches(["xx", "x", "xxx"], 2) == [[2, 0], [1]]
    assert length_sorted_batches([], 32) == []


# --------------------------------------------------------------------------- sparse


def test_bm25_tokens_keep_document_numbers_whole() -> None:
    text = "Invoice INV-2026-0417 for the PO: total 1,234.50 due 14/03/2026."

    assert bm25_tokens(text) == [
        "invoice",
        "inv-2026-0417",
        "po",
        "total",
        "1,234.50",
        "due",
        "14/03/2026",
    ]


def test_bm25_scores_rank_the_exact_match_first() -> None:
    texts = [chunk.text for chunk in fixture_chunks()]
    encoder = BM25Encoder().fit(texts)

    documents = encoder.encode_documents(texts)
    query = encoder.encode_query("PO-88231")
    scores = [dot(query, doc) for doc in documents]

    assert all(len(doc) > 0 for doc in documents)  # sparse encoding is never empty
    best = texts[scores.index(max(scores))]
    assert best.startswith("[PURCHASE ORDER PO-88231")
    assert math.isclose(sum(query.values), 1.0)


def test_rare_terms_weigh_more_in_queries() -> None:
    encoder = BM25Encoder().fit(["invoice total", "invoice courier", "invoice pallet wrap"])

    query = encoder.encode_query("invoice courier")
    weights = dict(zip(query.indices, query.values, strict=True))

    assert weights[token_index("courier")] > weights[token_index("invoice")]


def test_params_round_trip_through_json(tmp_path: Path) -> None:
    encoder = BM25Encoder(k1=1.5).fit(["invoice INV-1 total", "receipt kopi"])
    path = tmp_path / "processed" / "bm25.json"

    assert encoder.save(path) == path
    loaded = BM25Encoder.load(path)

    assert loaded.to_params() == encoder.to_params()
    assert loaded.encode_query("INV-1") == encoder.encode_query("INV-1")


def test_unfitted_encoder_and_bad_params_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        BM25Encoder().encode_query("x")
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Encoder().fit([])
    with pytest.raises(ValueError, match="unsupported BM25 params version"):
        BM25Encoder.from_params({"version": 99})


# --------------------------------------------------------------------------- chunks


def test_chunk_embedder_builds_pinecone_records() -> None:
    document_id = new_document_id()
    chunks = fixture_chunks()[:2]  # invoice header + line items
    embedder = ChunkEmbedder(
        DenseEmbedder(EmbeddingConfig(), model=FakeEncoder()),
        BM25Encoder().fit(c.text for c in chunks),
    )

    embedded = embedder.embed(document_id, chunks, metadata={"vendor": "Acme Supplies Ltd"})

    assert [e.id for e in embedded] == [f"{document_id}#c0", f"{document_id}#c1"]
    header = embedded[0].to_pinecone()
    assert set(header) == {"id", "values", "sparse_values", "metadata"}
    assert header["metadata"] == {
        "document_id": document_id,
        "chunk_index": 0,
        "doc_type": DocType.INVOICE.value,
        "kind": "header",
        "pages": ["1"],
        "text": chunks[0].text,
        "vendor": "Acme Supplies Ltd",
    }
    assert len(header["values"]) == 8


# --------------------------------------------------------------------------- real models


def test_tiny_sentence_transformer_end_to_end() -> None:
    pytest.importorskip("sentence_transformers")
    try:
        model = load_sentence_transformer(TINY_MODEL)
    except OSError as exc:  # offline and not cached
        pytest.skip(f"cannot load {TINY_MODEL}: {exc}")
    embedder = DenseEmbedder(EmbeddingConfig(dense_model=TINY_MODEL, batch_size=4), model=model)
    texts = [chunk.text for chunk in fixture_chunks()]

    vectors = np.array(embedder.embed_documents(texts))

    assert vectors.shape == (len(texts), embedder.dimension) == (len(texts), 128)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    assert load_sentence_transformer(TINY_MODEL) is model  # cached


def test_bge_base_is_768_dimensional_and_normalized() -> None:
    pytest.importorskip("sentence_transformers")
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache("BAAI/bge-base-en-v1.5", "config.json"), str):
        pytest.skip("bge-base-en-v1.5 not downloaded (440 MB); runs where it is cached")
    embedder = DenseEmbedder()

    query = embedder.embed_query("which invoice billed PO-88231?")
    vectors = embedder.embed_documents([c.text for c in fixture_chunks()])

    assert embedder.dimension == 768 and len(query) == 768
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
