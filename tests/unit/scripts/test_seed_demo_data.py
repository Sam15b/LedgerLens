"""Demo bundles: the seed selection carried to a cloud environment as one archive."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

import scripts.seed_demo_data as seed
from ledgerlens.storage.bulk_load import Rendered, read_manifest


def _ledger(root: Path) -> tuple[list[dict[str, Any]], list[Rendered]]:
    (root / "pdf" / "invoice").mkdir(parents=True)
    pdf = root / "pdf" / "invoice" / "SINV-1.pdf"
    pdf.write_bytes(b"%PDF-1.7 demo")
    demo: list[dict[str, Any]] = [
        {"id": "V-1", "record_type": "vendor"},
        {
            "id": "SINV-1",
            "record_type": "invoice",
            "vendor_id": "V-1",
            "payment_status": "unpaid",
            "anomalies": [],
            "data": {
                "vendor_name": "Demo Vendor",
                "invoice_number": "INV-1",
                "currency": "EUR",
                "total": "10.00",
            },
        },
    ]
    return demo, [Rendered("SINV-1", "invoice", pdf)]


def test_bundle_round_trip(tmp_path: Path) -> None:
    demo, rendered = _ledger(tmp_path / "synthetic")
    archive = seed.export_bundle(demo, rendered, tmp_path / "synthetic", tmp_path / "b.tar.gz")

    root = seed.fetch_bundle(str(archive), tmp_path / "work")

    records = (root / seed.BUNDLE_RECORDS).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in records] == demo
    (doc,) = read_manifest(root / seed.BUNDLE_MANIFEST)
    assert (doc.record_id, doc.doc_type) == ("SINV-1", "invoice")
    assert doc.path == root / "pdf" / "invoice" / "SINV-1.pdf"
    assert doc.path.read_bytes() == b"%PDF-1.7 demo"


def test_s3_bundles_are_downloaded_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo, rendered = _ledger(tmp_path / "synthetic")
    archive = seed.export_bundle(demo, rendered, tmp_path / "synthetic", tmp_path / "b.tar.gz")
    fetched: list[tuple[str, str]] = []

    def fake_download(bucket: str, key: str, dest: Path) -> None:
        fetched.append((bucket, key))
        dest.write_bytes(archive.read_bytes())

    monkeypatch.setattr(seed, "_download_s3", fake_download)
    root = seed.fetch_bundle("s3://demo-bucket/demo/b.tar.gz", tmp_path / "work")

    assert fetched == [("demo-bucket", "demo/b.tar.gz")]
    assert (root / seed.BUNDLE_RECORDS).is_file()


@pytest.mark.parametrize("uri", ["s3://bucket", "s3:///key", "https://bucket/key", "bucket/key"])
def test_bad_s3_uris_are_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="s3://bucket/key"):
        seed.split_s3_uri(uri)


def _tar_with(tmp_path: Path, name: str) -> Path:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    return archive


def test_archive_members_cannot_escape_the_workdir(tmp_path: Path) -> None:
    archive = _tar_with(tmp_path, "../escaped.jsonl")

    with pytest.raises(tarfile.TarError):
        seed.fetch_bundle(str(archive), tmp_path / "work")
    assert not (tmp_path / "escaped.jsonl").exists()


def test_an_archive_without_records_is_not_a_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a demo bundle"):
        seed.fetch_bundle(str(_tar_with(tmp_path, "other.json")), tmp_path / "work")


def test_main_exports_then_dry_runs_the_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "synthetic"
    demo, rendered = _ledger(root)
    (root / "records.jsonl").write_text("", encoding="utf-8")
    bundle = seed.export_bundle(demo, rendered, root, tmp_path / "b.tar.gz")

    assert seed.main(["--bundle", str(bundle), "--dry-run"]) == 0
    assert "demo: {'vendor': 1, 'invoice': 1}; 1 rendered documents" in capsys.readouterr().out


def test_unreadable_bundle_is_exit_code_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert seed.main(["--bundle", str(tmp_path / "missing.tar.gz")]) == 1
    assert "cannot read the bundle" in capsys.readouterr().out
