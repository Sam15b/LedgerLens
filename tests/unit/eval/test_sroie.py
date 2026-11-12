from pathlib import Path

import pytest

from eval.datasets.sroie import iter_sroie, parse_box_file, parse_entities, read_text

BOX = """\
72,25,326,25,326,64,72,64,TAN WOON YANN
50,82,440,82,440,121,50,121,BOOK TA .K(TAMAN DAYA) SDN BND
110,312,214,312,214,334,110,334,TOTAL: 9.00, INCL. GST
not,a,box
"""
ENTITIES = '{"company": "BOOK TA .K(TAMAN DAYA) SDN BHD", "date": "25/12/2018", "total": "9.00"}'


def write_sample(base: Path, sample_id: str, *, entities: bool = True) -> None:
    (base / "img").mkdir(parents=True, exist_ok=True)
    (base / "box").mkdir(exist_ok=True)
    (base / "img" / f"{sample_id}.jpg").write_bytes(b"\xff\xd8\xff")
    (base / "box" / f"{sample_id}.txt").write_text(BOX, encoding="utf-8")
    if entities:
        (base / "entities").mkdir(exist_ok=True)
        (base / "entities" / f"{sample_id}.txt").write_text(ENTITIES, encoding="utf-8")


def test_box_lines_keep_commas_in_text_and_skip_malformed_rows() -> None:
    lines = parse_box_file(BOX)

    assert [line.text for line in lines] == [
        "TAN WOON YANN",
        "BOOK TA .K(TAMAN DAYA) SDN BND",
        "TOTAL: 9.00, INCL. GST",
    ]
    assert lines[0].polygon == ((72, 25), (326, 25), (326, 64), (72, 64))


def test_entities_keep_known_fields_only() -> None:
    assert parse_entities(ENTITIES) == {
        "company": "BOOK TA .K(TAMAN DAYA) SDN BHD",
        "date": "25/12/2018",
        "total": "9.00",
    }
    assert parse_entities("not json") == {}


def test_latin1_files_are_read(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes("CAF\xc9".encode("latin-1"))

    assert read_text(path) == "CAFÉ"


def test_split_layout_is_sorted_and_limited(tmp_path: Path) -> None:
    for sample_id in ("X003", "X001", "X002"):
        write_sample(tmp_path / "test", sample_id)
    (tmp_path / "test" / "img" / "X004.jpg").write_bytes(b"")  # no box file: skipped

    samples = list(iter_sroie(tmp_path, "test", limit=2))

    assert [s.sample_id for s in samples] == ["X001", "X002"]
    assert samples[0].fields["total"] == "9.00"
    assert samples[0].text.splitlines()[0] == "TAN WOON YANN"


def test_flat_layout_has_no_fields(tmp_path: Path) -> None:
    (tmp_path / "X1.jpg").write_bytes(b"")
    (tmp_path / "X1.txt").write_text(BOX, encoding="utf-8")

    (sample,) = iter_sroie(tmp_path)

    assert sample.image_path == tmp_path / "X1.jpg" and sample.fields == {}


def test_missing_dataset_explains_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rrc.cvc.uab.es"):
        list(iter_sroie(tmp_path / "sroie"))
