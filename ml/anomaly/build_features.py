"""Build the anomaly model's train / val / test feature matrices from the synthetic ledger.

    python -m ml.anomaly.build_features --records data/synthetic/records.jsonl \\
        --out data/processed/anomaly [--embeddings header_embeddings.npz]

Every invoice in ``records.jsonl`` becomes one row of :func:`ledgerlens.anomaly.features
.invoice_features`, computed in arrival order (invoice date, then record id) so a vendor's
history only holds invoices that came before. Labels come from the ledger: the injected
anomaly type, or ``normal``.

The split is by time: the earliest 70 % of invoices train, the next 15 % validate, the
last 15 % test. A model has to score invoices that arrive after the ones it learned from,
so the split does too. Every split keeps its anomalies; the autoencoder trains on the
``normal`` rows of ``train`` and picks its threshold on the labelled ``val``.

The scaler (:class:`~ledgerlens.anomaly.features.Standardizer`) is fit on the training
split's normal rows only (the rows the autoencoder learns "normal" from) and applied to
all three. ``--embeddings`` adds the 16 PCA components of each invoice's header-chunk
embedding: an ``.npz`` with ``ids`` (invoice record ids) and ``vectors``. The PCA is fit
on the training rows too, and invoices without an embedding get zeros.

Output in ``--out``: ``train.npz`` / ``val.npz`` / ``test.npz`` (``x`` scaled, ``x_raw``,
``ids``, ``labels``), ``scaler.json``, ``pca.json`` (with embeddings) and
``manifest.json`` (feature names, rows per split and label, date ranges, source hash).
Building fails if any value is NaN or inf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ledgerlens.anomaly.features import (
    EmbeddingPCA,
    HistoryTracker,
    Matrix,
    Standardizer,
    feature_names,
    feature_vector,
    invoice_features,
)
from ledgerlens.core.schemas import Invoice
from ledgerlens.extraction.normalize import canonical_vendor

logger = logging.getLogger(__name__)

NORMAL = "normal"
SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class LabelledInvoice:
    id: str
    vendor_key: str
    """``vendor_id``, or the canonical name for a vendor the master does not know."""
    invoice: Invoice
    label: str


def load_invoices(path: Path) -> list[LabelledInvoice]:
    """The ledger's invoices with their labels, in arrival order (date, then id)."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "invoice":
                continue
            invoice = Invoice.model_validate(record["data"])
            vendor = record.get("vendor_id") or f"name:{canonical_vendor(invoice.vendor_name)}"
            anomalies = record.get("anomalies") or []
            rows.append(
                LabelledInvoice(
                    record["id"], vendor, invoice, anomalies[0] if anomalies else NORMAL
                )
            )
    return sorted(rows, key=lambda r: (r.invoice.invoice_date, r.id))


def feature_rows(invoices: Iterable[LabelledInvoice]) -> list[dict[str, float]]:
    """Each invoice's features, with its vendor's history up to it (``invoices`` must be in
    arrival order)."""
    tracker = HistoryTracker()
    rows = []
    for item in invoices:
        rows.append(invoice_features(item.invoice, tracker.history(item.vendor_key)))
        tracker.add(item.vendor_key, item.invoice.invoice_date, item.invoice.total)
    return rows


def split_by_time(n: int, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> dict[str, slice]:
    """Consecutive slices of ``n`` arrival-ordered rows."""
    if len(fractions) != len(SPLITS) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"need {len(SPLITS)} fractions summing to 1, got {fractions}")
    train_end = round(n * fractions[0])
    val_end = train_end + round(n * fractions[1])
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def load_embeddings(path: Path) -> dict[str, Matrix]:
    with np.load(path, allow_pickle=False) as data:
        return {
            str(i): np.asarray(v, dtype=np.float64)
            for i, v in zip(data["ids"], data["vectors"], strict=True)
        }


def embedding_matrix(
    ids: Sequence[str], vectors: Mapping[str, Matrix], train: slice
) -> tuple[Matrix, EmbeddingPCA]:
    """The PCA components of each row's embedding (zeros where there is none), with the
    PCA fit on the training rows that have one."""
    dim = len(next(iter(vectors.values())))
    fit_on = [vectors[i] for i in ids[train] if i in vectors]
    pca = EmbeddingPCA.fit(np.stack(fit_on))
    out = np.zeros((len(ids), pca.components.shape[0]))
    have = [k for k, i in enumerate(ids) if i in vectors]
    if have:
        out[have] = pca.transform(np.stack([vectors[ids[k]] for k in have]))
    logger.info(
        "embeddings: %d of %d invoices (%d-d), PCA fit on %d", len(have), len(ids), dim, len(fit_on)
    )
    return out, pca


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    records: Path,
    out: Path,
    *,
    embeddings: Path | None = None,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    """Write the matrices, scaler and manifest to ``out``; return the manifest."""
    invoices = load_invoices(records)
    if not invoices:
        raise ValueError(f"no invoices in {records}")
    ids = [item.id for item in invoices]
    labels = np.array([item.label for item in invoices])
    splits = split_by_time(len(invoices), fractions)
    names = feature_names(embedding=embeddings is not None)
    x_raw = np.array([feature_vector(row) for row in feature_rows(invoices)], dtype=np.float64)
    pca = None
    if embeddings is not None:
        emb, pca = embedding_matrix(ids, load_embeddings(embeddings), splits["train"])
        x_raw = np.hstack([x_raw, emb])
    if not np.isfinite(x_raw).all():
        bad = sorted({names[c] for c in np.argwhere(~np.isfinite(x_raw))[:, 1]})
        raise ValueError(f"non-finite feature values in: {', '.join(bad)}")

    train_rows = np.arange(len(ids))[splits["train"]]
    normal_train = train_rows[labels[train_rows] == NORMAL]
    scaler = Standardizer.fit(x_raw[normal_train], names)
    x = scaler.transform(x_raw)

    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": str(records),
        "source_sha256": file_sha256(records),
        "features": list(names),
        "scaler_fit_on": f"train, label {NORMAL!r} ({len(normal_train)} rows)",
        "splits": {},
    }
    for split, rows in splits.items():
        np.savez_compressed(
            out / f"{split}.npz",
            x=x[rows],
            x_raw=x_raw[rows],
            ids=np.array(ids[rows]),
            labels=labels[rows],
        )
        dates = [item.invoice.invoice_date for item in invoices[rows]]
        manifest["splits"][split] = {
            "rows": len(dates),
            "from": min(dates).isoformat() if dates else None,
            "to": max(dates).isoformat() if dates else None,
            "labels": dict(sorted(Counter(labels[rows].tolist()).items())),
        }
    (out / "scaler.json").write_text(
        json.dumps(scaler.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    if pca is not None:
        (out / "pca.json").write_text(json.dumps(pca.to_json()) + "\n", encoding="utf-8")
        manifest["pca_explained_variance"] = round(float(pca.explained_variance_ratio.sum()), 4)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.anomaly.build_features",
        description="Build the anomaly autoencoder's feature matrices.",
    )
    parser.add_argument("--records", type=Path, default=Path("data/synthetic/records.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/anomaly"))
    parser.add_argument("--embeddings", type=Path, help=".npz with ids and vectors (optional)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manifest = build(args.records, args.out, embeddings=args.embeddings)
    for split, info in manifest["splits"].items():
        print(f"{split:5} {info['rows']:5} rows  {info['from']} .. {info['to']}  {info['labels']}")
    print(f"{len(manifest['features'])} features -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
