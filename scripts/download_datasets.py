"""Download the public datasets into data/raw/<name>/. Safe to run repeatedly.

    python scripts/download_datasets.py                # every dataset
    python scripts/download_datasets.py funsd cuad     # only these
    python scripts/download_datasets.py cord --force   # re-download

| name  | content                                   | source                         |
|-------|-------------------------------------------|--------------------------------|
| funsd | ~200 noisy scanned forms                  | zip from the FUNSD site        |
| cuad  | ~500 annotated commercial contracts (JSON)| zip from the CUAD GitHub repo  |
| cord  | ~1k receipts with line items (~2.3 GB)    | Hugging Face naver-clova-ix/cord-v2 |
| sroie | ~1k scanned receipts                      | manual: needs ICDAR registration |

A dataset whose folder already has files is skipped. data/raw/manifest.json records what
was fetched, from where, its license and size. Exit code 1 if any download failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "raw"
MANIFEST_NAME = "manifest.json"
IGNORED_FILES = {".gitkeep", ".DS_Store"}

Kind = Literal["zip", "huggingface", "manual"]


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: Kind
    source: str
    license: str
    description: str
    instructions: str = ""


SROIE_INSTRUCTIONS = """\
SROIE must be downloaded by hand (the ICDAR site requires a free account):
  1. Register / log in at https://rrc.cvc.uab.es/?ch=13
  2. Open the "Downloads" tab and download the task 1-3 training and test archives
  3. Extract them into {dest}
  4. Re-run this script to record SROIE in the manifest"""

DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset(
            "funsd",
            "zip",
            "https://guillaumejaume.github.io/FUNSD/dataset.zip",
            "Non-commercial research use (see https://guillaumejaume.github.io/FUNSD/)",
            "Noisy scanned forms with key-value annotations",
        ),
        Dataset(
            "cuad",
            "zip",
            "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
            "CC BY 4.0",
            "Commercial contracts with 41 clause categories (SQuAD-style JSON)",
        ),
        Dataset(
            "cord",
            "huggingface",
            "naver-clova-ix/cord-v2",
            "CC BY 4.0",
            "Receipts with line-item labels (parquet with images, ~2.3 GB)",
        ),
        Dataset(
            "sroie",
            "manual",
            "https://rrc.cvc.uab.es/?ch=13",
            "ICDAR 2019 SROIE competition terms",
            "Scanned receipts with 4 key fields",
            SROIE_INSTRUCTIONS,
        ),
    )
}


# --- helpers ------------------------------------------------------------------------------


def dataset_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.name not in IGNORED_FILES]


def is_present(folder: Path) -> bool:
    return bool(dataset_files(folder))


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing members that would land outside ``dest`` (zip slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in archive: {member.filename!r}")
        zf.extractall(dest)


def flatten_single_root(dest: Path) -> None:
    """If extraction produced exactly one top-level folder, move its contents up."""
    entries = [p for p in dest.iterdir() if p.name not in IGNORED_FILES | {"__MACOSX"}]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), dest / child.name)
        inner.rmdir()
    shutil.rmtree(dest / "__MACOSX", ignore_errors=True)


def download_zip(url: str, dest: Path, *, client: httpx.Client) -> str:
    """Stream ``url`` to a temp file, extract into ``dest``; return the archive's SHA-256."""
    dest.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "download.zip"
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in response.iter_bytes(1 << 20):
                    fh.write(chunk)
                    digest.update(chunk)
        safe_extract(archive, dest)
    flatten_single_root(dest)
    return digest.hexdigest()


def download_huggingface(repo_id: str, dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("huggingface_hub is required: pip install huggingface_hub") from exc
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest)


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def summarize(folder: Path) -> dict[str, int]:
    files = dataset_files(folder)
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}


# --- main --------------------------------------------------------------------------------


def fetch(dataset: Dataset, dest: Path, *, force: bool, client: httpx.Client) -> dict[str, Any]:
    """Fetch one dataset; returns its manifest entry (``status`` says what happened)."""
    folder = dest / dataset.name
    entry: dict[str, Any] = {
        "source": dataset.source,
        "license": dataset.license,
        "description": dataset.description,
    }

    if is_present(folder) and not force:
        return entry | {"status": "present"} | summarize(folder)

    if dataset.kind == "manual":
        print(dataset.instructions.format(dest=folder))
        return entry | {"status": "manual_download_required"}

    if force and folder.exists():
        shutil.rmtree(folder)
    print(f"[{dataset.name}] downloading {dataset.source} ...", flush=True)
    if dataset.kind == "zip":
        entry["sha256"] = download_zip(dataset.source, folder, client=client)
    else:
        download_huggingface(dataset.source, folder)
    return entry | {"status": "downloaded"} | summarize(folder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="dataset", help=", ".join(DATASETS))
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    names: list[str] = args.names or list(DATASETS)
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        parser.error(
            f"unknown dataset(s): {', '.join(unknown)} (choose from {', '.join(DATASETS)})"
        )
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    manifest_path = dest / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    failed: list[str] = []
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        for name in names:
            try:
                entry = fetch(DATASETS[name], dest, force=args.force, client=client)
            except Exception as exc:  # keep going; report every failure at the end
                failed.append(name)
                entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            entry["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            manifest[name] = manifest.get(name, {}) | entry
            size = ""
            if "files" in entry:
                size = f", {entry['files']} files, {entry['bytes'] / 1e6:.1f} MB"
            print(f"[{name}] {entry['status']}{size}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if failed:
        print(f"failed: {', '.join(failed)} (see {manifest_path})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
