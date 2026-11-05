"""Tests for scripts/download_datasets.py (loaded by path: scripts/ is not a package)."""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "download_datasets.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


dd = _load()


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def _client(payload: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    )


def test_is_present_ignores_placeholder_files(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").write_text("")
    assert not dd.is_present(tmp_path)
    assert not dd.is_present(tmp_path / "missing")

    (tmp_path / "a.json").write_text("{}")
    assert dd.is_present(tmp_path)


def test_download_zip_extracts_and_flattens_single_root(tmp_path: Path) -> None:
    payload = _zip_bytes({"dataset/train/a.json": "{}", "dataset/test/b.json": "{}"})

    digest = dd.download_zip("https://x/data.zip", tmp_path / "funsd", client=_client(payload))

    assert (tmp_path / "funsd" / "train" / "a.json").is_file()
    assert (tmp_path / "funsd" / "test" / "b.json").is_file()
    assert len(digest) == 64
    assert sorted(p.name for p in tmp_path.iterdir()) == ["funsd"]  # temp archive cleaned up


def test_download_zip_flattens_root_containing_its_own_name(tmp_path: Path) -> None:
    payload = _zip_bytes({"data/data/a.json": "{}", "data/b.json": "{}"})

    dd.download_zip("https://x/data.zip", tmp_path / "cuad", client=_client(payload))

    assert sorted(
        p.relative_to(tmp_path / "cuad").as_posix() for p in dd.dataset_files(tmp_path / "cuad")
    ) == ["b.json", "data/a.json"]
    assert sorted(p.name for p in (tmp_path / "cuad").iterdir()) == ["b.json", "data"]


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../escape.txt": "x"}))

    with pytest.raises(ValueError, match="unsafe path"):
        dd.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_main_skips_present_and_records_manifest(tmp_path: Path) -> None:
    (tmp_path / "cuad").mkdir()
    (tmp_path / "cuad" / "CUADv1.json").write_text("{}")

    assert dd.main(["cuad", "sroie", "--dest", str(tmp_path)]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cuad"]["status"] == "present"
    assert manifest["cuad"]["files"] == 1
    assert manifest["cuad"]["license"] == "CC BY 4.0"
    assert manifest["sroie"]["status"] == "manual_download_required"


def test_main_reports_failures_and_keeps_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> str:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(dd, "download_zip", boom)
    (tmp_path / "cuad").mkdir()
    (tmp_path / "cuad" / "x.json").write_text("{}")

    assert dd.main(["funsd", "cuad", "--dest", str(tmp_path)]) == 1

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["funsd"]["status"] == "failed"
    assert "offline" in manifest["funsd"]["error"]
    assert manifest["cuad"]["status"] == "present"


def test_main_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        dd.main(["nope", "--dest", str(tmp_path)])


def test_failed_download_leaves_no_dataset_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def partial_then_fail(repo_id: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "train-00000.parquet").write_text("half")
        raise httpx.ReadError("connection reset")

    monkeypatch.setattr(dd, "download_huggingface", partial_then_fail)

    assert dd.main(["cord", "--dest", str(tmp_path)]) == 1

    assert not dd.is_present(tmp_path / "cord")  # a rerun will not skip it
    assert (tmp_path / "cord.partial" / "train-00000.parquet").is_file()  # kept for resume


def test_transient_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def flaky(repo_id: str, dest: Path) -> None:
        calls.append(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            raise httpx.ReadError("connection reset")
        (dest / "data.parquet").write_text("ok")
        (dest / ".cache").mkdir()
        (dest / ".cache" / "meta.json").write_text("{}")

    monkeypatch.setattr(dd, "download_huggingface", flaky)

    assert dd.main(["cord", "--dest", str(tmp_path)]) == 0

    assert len(calls) == 2
    assert (tmp_path / "cord" / "data.parquet").is_file()
    assert not (tmp_path / "cord.partial").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cord"] | {"status": "downloaded", "files": 1} == manifest["cord"]
