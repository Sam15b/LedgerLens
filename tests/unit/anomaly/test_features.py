"""Anomaly features: values per group, vendor history, PCA, scaling and the batch build."""

import json
import math
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ledgerlens.anomaly.features import (
    EMBEDDING_DIM,
    EMBEDDING_FEATURES,
    FEATURE_NAMES,
    EmbeddingPCA,
    HistoryTracker,
    Standardizer,
    VendorHistory,
    feature_names,
    feature_vector,
    invoice_features,
    is_round,
    signed_log,
    significant_digits,
)
from ledgerlens.core.schemas import Invoice
from ml.anomaly.build_features import build, load_invoices, split_by_time


def invoice(**overrides: Any) -> Invoice:
    data: dict[str, Any] = {
        "invoice_number": "INV-1",
        "vendor_name": "Acme Supplies Ltd",
        "vendor_tax_id": "GB123456789",
        "invoice_date": "2026-03-04",  # a Wednesday
        "due_date": "2026-04-03",
        "po_number": "PO-1",
        "currency": "GBP",
        "subtotal": "200.00",
        "tax": "40.00",
        "total": "240.00",
        "line_items": [
            {"description": "Boxes", "quantity": "10", "unit_price": "15.00", "amount": "150.00"},
            {"description": "Tape", "quantity": "5", "amount": "50.00"},
        ],
    }
    return Invoice.model_validate(data | overrides)


# --------------------------------------------------------------------------- amount


def test_signed_log_and_significant_digits() -> None:
    assert signed_log(0) == 0 and signed_log(Decimal("-9")) == -math.log(10)
    assert [significant_digits(Decimal(v)) for v in ("1000", "1500", "285.90", "0.50")] == [
        1,
        2,
        4,
        1,
    ]
    assert is_round(Decimal("1000")) and is_round(Decimal("15000000"))
    assert not is_round(Decimal("285.90")) and not is_round(Decimal("0"))
    assert not is_round(Decimal("240.00"))  # small totals often have two digits


def test_amount_features() -> None:
    values = invoice_features(invoice())
    assert values["log_total"] == pytest.approx(math.log1p(240))
    assert values["tax_ratio"] == pytest.approx(0.2)
    assert values["round_total"] == 0.0
    assert values["total_zscore"] == 0.0  # no history

    # no subtotal: tax over total - tax; no tax: 0
    assert invoice_features(invoice(subtotal=None))["tax_ratio"] == pytest.approx(0.2)
    assert invoice_features(invoice(tax=None))["tax_ratio"] == 0.0
    rounded = invoice(total="1000", subtotal=None, tax=None, line_items=[])
    assert invoice_features(rounded)["round_total"] == 1.0


def test_zscore_against_vendor_history() -> None:
    days = [date(2026, 1, d) for d in (5, 12, 19)]
    history = VendorHistory.from_invoices(
        zip(days, map(Decimal, ("200", "240", "280")), strict=True)
    )
    assert history.count == 3 and history.last_invoice_date == date(2026, 1, 19)

    typical = invoice_features(invoice(), history)["total_zscore"]
    inflated = invoice_features(invoice(total="2400.00", subtotal=None), history)["total_zscore"]
    assert abs(typical) < 1 < inflated
    assert inflated <= 10  # clipped

    # too little history, or no spread at all, does not blow up
    one = VendorHistory.from_invoices([(days[0], Decimal("240"))])
    assert one.zscore(9.0) == 0.0
    same = VendorHistory.from_invoices([(d, Decimal("240")) for d in days])
    assert same.zscore(math.log1p(240)) == pytest.approx(0) and same.zscore(math.log1p(250)) < 10


# --------------------------------------------------------------------------- timing


def test_timing_features() -> None:
    history = VendorHistory.from_invoices([(date(2026, 2, 25), Decimal("100"))])
    values = invoice_features(invoice(), history)

    assert values["days_to_due"] == 30 and values["has_due_date"] == 1
    assert (values["weekday"], values["is_weekend"], values["day_of_month"]) == (2, 0, 4)
    assert values["log_days_since_last"] == pytest.approx(math.log1p(7))
    assert values["first_invoice"] == 0

    rush = invoice_features(invoice(invoice_date="2026-03-07", due_date="2026-03-08"))
    assert (rush["is_weekend"], rush["days_to_due"], rush["first_invoice"]) == (1, 1, 1)
    no_due = invoice_features(invoice(due_date=None))
    assert (no_due["days_to_due"], no_due["has_due_date"]) == (0, 0)


# --------------------------------------------------------------------------- structure, vendor


def test_structure_and_vendor_features() -> None:
    history = VendorHistory.from_invoices(
        [(date(2026, 1, 5), Decimal("100")), (date(2026, 2, 5), Decimal("300"))]
    )
    values = invoice_features(invoice(), history)

    assert values["line_count"] == 2
    # unit prices 15.00 and 50 / 5 = 10.00 (no unit price given)
    assert values["log_mean_unit_price"] == pytest.approx((math.log1p(15) + math.log1p(10)) / 2)
    assert values["log_max_unit_price"] == pytest.approx(math.log1p(15))
    assert (values["has_po"], values["has_tax_id"]) == (1, 1)
    assert values["log_vendor_invoice_count"] == pytest.approx(math.log(3))
    assert values["vendor_mean_log_total"] == pytest.approx(history.mean_log_total)
    assert values["vendor_std_log_total"] == pytest.approx(history.std_log_total)

    bare = invoice_features(invoice(po_number=None, vendor_tax_id=None, line_items=[]))
    assert [bare[k] for k in ("line_count", "log_max_unit_price", "has_po", "has_tax_id")] == [
        0,
        0,
        0,
        0,
    ]


def test_every_feature_is_finite_and_ordered() -> None:
    values = invoice_features(invoice(total="-50.00", subtotal=None, tax=None))
    assert tuple(values) == FEATURE_NAMES
    assert all(math.isfinite(v) for v in feature_vector(values))


def test_history_tracker_counts_only_earlier_invoices() -> None:
    tracker = HistoryTracker()
    assert tracker.history("V1").count == 0
    tracker.add("V1", date(2026, 1, 1), Decimal("100"))
    tracker.add("V2", date(2026, 1, 2), Decimal("5"))
    tracker.add("V1", date(2026, 1, 9), Decimal("120"))

    history = tracker.history("V1")
    assert history.count == 2 and history.last_invoice_date == date(2026, 1, 9)
    assert tracker.history("V2").count == 1


# --------------------------------------------------------------------------- PCA, scaling


def test_embedding_pca_round_trips_and_keeps_its_width() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(40, 32))
    pca = EmbeddingPCA.fit(embeddings)

    reduced = pca.transform(embeddings)
    assert reduced.shape == (40, EMBEDDING_DIM)
    assert np.allclose(reduced.mean(axis=0), 0, atol=1e-9)  # centred on the fit data
    ratios = pca.explained_variance_ratio
    assert np.all(np.diff(ratios) <= 1e-12) and 0 < ratios.sum() <= 1
    again = EmbeddingPCA.from_json(json.loads(json.dumps(pca.to_json())))
    assert np.allclose(again.transform(embeddings), reduced)
    assert list(pca.features(embeddings[0])) == list(EMBEDDING_FEATURES)

    small = EmbeddingPCA.fit(embeddings[:5])  # 5 samples: 5 components, the rest zero
    assert small.transform(embeddings).shape == (40, EMBEDDING_DIM)
    assert np.all(small.components[5:] == 0)
    with pytest.raises(ValueError, match="32-d"):
        pca.transform(np.zeros(8))
    with pytest.raises(ValueError, match="at least 2 rows"):
        EmbeddingPCA.fit(embeddings[:1])
    assert feature_names(embedding=True) == (*FEATURE_NAMES, *EMBEDDING_FEATURES)


def test_standardizer() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    scaler = Standardizer.fit(x, ["a", "const"])

    assert np.allclose(scaler.transform(x), [[-1, 0], [1, 0]])
    # a constant training column is only centred
    assert np.allclose(scaler.transform([[2.0, 7.0]]), [[0, 2]])
    # float noise in a constant column is not taken for spread
    noisy = Standardizer.fit([[0.1 + 0.2], [0.3], [0.1 + 0.2]], ["c"])
    assert noisy.scale[0] == 1.0
    again = Standardizer.from_json(json.loads(json.dumps(scaler.to_json())))
    assert again.names == ("a", "const") and np.allclose(again.transform(x), scaler.transform(x))
    with pytest.raises(ValueError, match="NaN or inf"):
        Standardizer.fit([[np.nan, 1.0]], ["a", "b"])
    with pytest.raises(ValueError, match="columns"):
        scaler.transform([[1.0]])


# --------------------------------------------------------------------------- batch build


def write_ledger(path: Path, n: int = 40) -> None:
    lines = [json.dumps({"record_type": "vendor", "id": "V1", "data": {}})]
    start = date(2026, 1, 1)
    for i in range(n):
        label = ["round_amount"] if i % 10 == 9 else []
        data = invoice(
            invoice_number=f"INV-{i}",
            invoice_date=(start + timedelta(days=i)).isoformat(),
            due_date=(start + timedelta(days=i + 30)).isoformat(),
            total="1000" if label else f"{240 + i}.00",
            subtotal=None,
        ).model_dump(mode="json")
        vendor = "V1" if i % 2 else None  # every other invoice from an unknown vendor
        record = {"record_type": "invoice", "id": f"S{n - i:03d}", "vendor_id": vendor}
        lines.append(json.dumps(record | {"anomalies": label, "data": data}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_invoices_load_in_arrival_order_with_labels(tmp_path: Path) -> None:
    write_ledger(tmp_path / "records.jsonl")
    invoices = load_invoices(tmp_path / "records.jsonl")

    dates = [item.invoice.invoice_date for item in invoices]
    assert dates == sorted(dates) and len(invoices) == 40  # ids run backwards: sorted by date
    assert {item.label for item in invoices} == {"normal", "round_amount"}
    assert invoices[0].vendor_key == "name:acme supplies"
    assert invoices[1].vendor_key == "V1"


def test_split_by_time() -> None:
    assert split_by_time(20) == {"train": slice(0, 14), "val": slice(14, 17), "test": slice(17, 20)}
    with pytest.raises(ValueError):
        split_by_time(10, (0.5, 0.5))


def test_build_writes_finite_scaled_matrices(tmp_path: Path) -> None:
    write_ledger(tmp_path / "records.jsonl")
    rng = np.random.default_rng(1)
    np.savez(
        tmp_path / "emb.npz",
        ids=np.array([f"S{i:03d}" for i in range(1, 31)]),  # 10 invoices have none
        vectors=rng.normal(size=(30, 24)),
    )

    manifest = build(tmp_path / "records.jsonl", tmp_path / "out", embeddings=tmp_path / "emb.npz")

    names = feature_names(embedding=True)
    assert manifest["features"] == list(names)
    assert {s: manifest["splits"][s]["rows"] for s in ("train", "val", "test")} == {
        "train": 28,
        "val": 6,
        "test": 6,
    }
    train = np.load(tmp_path / "out" / "train.npz")
    assert train["x"].shape == (28, len(names)) and np.isfinite(train["x"]).all()
    # the scaler is fit on the normal training rows: they come out centred
    normal = train["x"][train["labels"] == "normal"]
    assert np.allclose(normal.mean(axis=0), 0, atol=1e-9)
    scaler = Standardizer.from_json(json.loads((tmp_path / "out" / "scaler.json").read_text()))
    test = np.load(tmp_path / "out" / "test.npz")
    assert np.allclose(scaler.transform(test["x_raw"]), test["x"])
    assert test["ids"].dtype.kind == "U" and set(test["labels"]) <= {"normal", "round_amount"}
    assert (tmp_path / "out" / "pca.json").exists()

    plain = build(tmp_path / "records.jsonl", tmp_path / "plain")
    assert plain["features"] == list(FEATURE_NAMES)
    assert plain["splits"]["train"]["labels"] == {"normal": 26, "round_amount": 2}
