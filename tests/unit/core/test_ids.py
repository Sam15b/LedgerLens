import pytest

from ledgerlens.core.ids import (
    chunk_vector_id,
    is_document_id,
    new_document_id,
    new_request_id,
    parse_vector_id,
    vector_id_prefix,
)


def test_new_document_ids_are_unique_canonical_uuids() -> None:
    ids = {new_document_id() for _ in range(100)}

    assert len(ids) == 100
    assert all(is_document_id(i) for i in ids)


def test_vector_id_round_trip() -> None:
    doc_id = new_document_id()

    vector_id = chunk_vector_id(doc_id, 7)

    assert vector_id == f"{doc_id}#c7"
    assert vector_id.startswith(vector_id_prefix(doc_id))
    assert parse_vector_id(vector_id) == (doc_id, 7)


@pytest.mark.parametrize("bad_doc_id", ["", "abc", "INV-889", new_document_id().upper()])
def test_chunk_vector_id_rejects_non_canonical_document_ids(bad_doc_id: str) -> None:
    with pytest.raises(ValueError):
        chunk_vector_id(bad_doc_id, 0)


def test_chunk_vector_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        chunk_vector_id(new_document_id(), -1)


@pytest.mark.parametrize(
    "bad_vector_id",
    ["", "abc#c1", f"{new_document_id()}#c", f"{new_document_id()}#x1", f"{new_document_id()}"],
)
def test_parse_vector_id_rejects_malformed_ids(bad_vector_id: str) -> None:
    with pytest.raises(ValueError):
        parse_vector_id(bad_vector_id)


def test_request_ids_are_hex() -> None:
    request_id = new_request_id()

    assert len(request_id) == 32
    int(request_id, 16)
