"""One anomaly score per invoice from the rules and the autoencoder (layer 8, serving).

The two methods catch different things (``eval/reports/2026-11-20/anomaly.md``):

* the **rules** (:mod:`ledgerlens.anomaly.rules`) are deterministic and precise: a
  duplicate, a breached contract rate or an unknown vendor is a fact, not a guess. Any rule
  hit is a **hard hit** and flags the invoice whatever the autoencoder says;
* the **autoencoder** catches what no rule describes (round amounts, odd shapes for a
  vendor). It flags an invoice whose reconstruction error reaches its threshold.

:func:`combine` puts both on one 0..1 scale so a review list can be sorted by it. With
``a = error / (error + threshold)`` (0.5 exactly at the threshold):

* hard hit: ``score = 0.5 + a / 2``, in [0.5, 1);
* otherwise ``score = a / 2``, in [0, 0.5).

So every rule hit ranks above every invoice flagged by the autoencoder alone, and the
autoencoder orders invoices within each band. ``flagged`` is a hard hit or ``error >=
threshold`` (``score >= 0.25`` without a hit).

Reasons (:func:`explain`): each rule hit's own sentence; when the autoencoder flags the
invoice, its three worst-reconstructed features in plain language ("total unusually high
for this vendor (z = +3.4)"); and for an invoice nothing flagged, one line saying so.

Serving: :class:`AnomalyScorer` holds the loaded model (:class:`Autoencoder`) and scores
one invoice at a time. :func:`get_scorer` loads it once per process (a worker scores many
invoices with one model), from ``configs/anomaly.yaml``. Without a saved model, or without
Keras installed, the scorer runs the rules alone and says so in its ``model_version``
(``rules-only``).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from ledgerlens.anomaly.features import (
    FEATURE_NAMES,
    Standardizer,
    VendorHistory,
    feature_vector,
    invoice_features,
    load_vendor_history,
)
from ledgerlens.anomaly.rules import (
    InvoiceRef,
    RuleContext,
    RuleHit,
    VendorMaster,
    check_invoice,
)
from ledgerlens.core.config import AnomalyConfig, get_config, repo_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

Matrix = npt.NDArray[np.float64]
Flags = npt.NDArray[np.bool_]

logger = logging.getLogger(__name__)

RULE_BAND = 0.5
"""Hard hits score at least this; invoices without one score below it."""
RULES_ONLY = "rules-only"
"""``model_version`` of scores made without an autoencoder."""


def ae_score(error: float | None, threshold: float) -> float:
    """``error / (error + threshold)``: 0 for a perfect reconstruction, 0.5 at the
    threshold, towards 1 far above it. 0 without an error (no model)."""
    if error is None or error <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    return float(error / (error + threshold))


@dataclass(frozen=True)
class FeatureError:
    """How badly the autoencoder reconstructed one feature of one invoice."""

    name: str
    value: float
    """The invoice's raw (unscaled) value."""
    error: float
    """Squared error on the scaled feature: its part of the reconstruction error."""
    above: bool
    """The invoice's value is higher than the model expected."""


@dataclass(frozen=True)
class AnomalyScore:
    """The combined verdict on one invoice."""

    score: float
    """0..1, for ranking: hard hits in [0.5, 1), the rest in [0, 0.5)."""
    flagged: bool
    ae_error: float | None
    """Reconstruction error (``None`` if the autoencoder was not run)."""
    ae_threshold: float
    rule_hits: tuple[RuleHit, ...] = ()
    feature_errors: tuple[FeatureError, ...] = ()
    """Per feature, largest error first (empty without the autoencoder)."""

    @property
    def hard_hit(self) -> bool:
        return bool(self.rule_hits)

    @property
    def ae_flagged(self) -> bool:
        return self.ae_error is not None and self.ae_error >= self.ae_threshold

    @property
    def reasons(self) -> list[Reason]:
        """:func:`explain`: plain-language reasons for this verdict."""
        return explain(self)


def combine(
    rule_hits: Sequence[RuleHit],
    error: float | None,
    threshold: float,
    feature_errors: Sequence[FeatureError] = (),
) -> AnomalyScore:
    """The rules' hits and the autoencoder's error on one invoice as one score."""
    a = ae_score(error, threshold)
    hit = bool(rule_hits)
    return AnomalyScore(
        score=RULE_BAND + a / 2 if hit else a / 2,
        flagged=hit or (error is not None and error >= threshold),
        ae_error=error,
        ae_threshold=threshold,
        rule_hits=tuple(rule_hits),
        feature_errors=tuple(feature_errors),
    )


def combine_arrays(hard_hits: Flags, errors: Matrix, threshold: float) -> tuple[Matrix, Flags]:
    """:func:`combine` over many invoices at once: ``(scores, flags)``."""
    hard_hits = np.asarray(hard_hits, dtype=bool)
    errors = np.asarray(errors, dtype=np.float64)
    a = np.where(errors > 0, errors / (errors + threshold), 0.0)
    scores: Matrix = np.where(hard_hits, RULE_BAND + a / 2, a / 2)
    flags: Flags = hard_hits | (errors >= threshold)
    return scores, flags


# --------------------------------------------------------------------------- the model


def import_keras() -> Any:
    """``keras``, on TensorFlow if it is installed, else PyTorch (unless ``KERAS_BACKEND``
    says otherwise). Must run before anything else imports keras."""
    if (
        "KERAS_BACKEND" not in os.environ
        and importlib.util.find_spec("tensorflow") is None
        and importlib.util.find_spec("torch") is not None
    ):
        os.environ["KERAS_BACKEND"] = "torch"
    import keras

    return keras


class ReconstructionModel(Protocol):
    """What the scorer needs from a model: :class:`Autoencoder`, or a stand-in in tests."""

    @property
    def version(self) -> str: ...
    @property
    def threshold(self) -> float: ...
    @property
    def features(self) -> tuple[str, ...]: ...

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        """``(scaled input, reconstruction)`` of unscaled feature rows."""
        ...


MODEL_FILE = "model.keras"


@dataclass(frozen=True)
class Autoencoder:
    """A saved ``models/anomaly/<version>/`` folder, loaded."""

    model: Any
    scaler: Standardizer
    version: str
    threshold: float
    """Flag at this reconstruction error: the rules + AE cut if one was saved (and wanted),
    else the autoencoder's own threshold."""
    features: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path, *, combined: bool = True) -> Autoencoder:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        scaler = Standardizer.from_json(
            json.loads((path / "scaler.json").read_text(encoding="utf-8"))
        )
        features = tuple(metadata["features"])
        if scaler.names != features:
            raise ValueError(f"{path}: scaler.json and metadata.json list different features")
        if features != FEATURE_NAMES:
            # the PCA text features need the header chunk's embedding at scoring time
            raise ValueError(f"{path}: the scorer computes {len(FEATURE_NAMES)} features only")
        saved = metadata.get("combined") if combined else None
        threshold = float((saved or metadata["threshold"])["value"])
        model = import_keras().saving.load_model(path / MODEL_FILE)
        return cls(model, scaler, str(metadata["version"]), threshold, features, metadata)

    def reconstruct(self, x_raw: Matrix) -> tuple[Matrix, Matrix]:
        x = self.scaler.transform(x_raw)
        return x, np.asarray(self.model.predict(x, verbose=0), dtype=np.float64)


def find_model(models_dir: Path, version: str = "latest") -> Path | None:
    """``models_dir/<version>``, or for ``latest`` the newest folder with a
    ``metadata.json`` (names are ``ae-<UTC time>``, so the last one sorted). ``None`` if
    there is none."""
    if version != "latest":
        path = models_dir / version
        return path if (path / "metadata.json").exists() else None
    if not models_dir.is_dir():
        return None
    found = sorted(p for p in models_dir.iterdir() if (p / "metadata.json").exists())
    return found[-1] if found else None


# --------------------------------------------------------------------------- reasons

TOP_FEATURES = 3

FEATURE_PHRASES: dict[str, tuple[str, str]] = {
    # feature: (the invoice's value is higher than expected, lower than expected)
    "log_total": ("total unusually high", "total unusually low"),
    "tax_ratio": ("tax rate higher than usual", "tax rate lower than usual"),
    "round_total": (
        "total is a round number, typed in rather than added up",
        "total less round than usual",
    ),
    # total_zscore is worded by the z-score's own sign (see describe_feature)
    "total_zscore": (
        "total higher than this vendor's usual",
        "total lower than this vendor's usual",
    ),
    "days_to_due": ("payment terms longer than usual", "payment terms unusually short"),
    "has_due_date": ("has a due date where similar invoices have none", "no due date"),
    "weekday": ("dated later in the week than usual", "dated earlier in the week than usual"),
    "is_weekend": ("dated on a weekend", "dated on a weekday where weekends are usual"),
    "day_of_month": (
        "dated later in the month than usual",
        "dated earlier in the month than usual",
    ),
    "log_days_since_last": (
        "long gap since this vendor's previous invoice",
        "arrived unusually soon after this vendor's previous invoice",
    ),
    "first_invoice": ("first invoice from this vendor", "vendor has invoiced before"),
    "line_count": ("more line items than usual", "fewer line items than usual"),
    "log_mean_unit_price": ("unit prices unusually high", "unit prices unusually low"),
    "log_max_unit_price": ("highest unit price unusually high", "highest unit price unusually low"),
    "has_po": ("cites a purchase order where similar invoices do not", "no purchase order cited"),
    "has_tax_id": ("carries a tax id where similar invoices do not", "no vendor tax id"),
    "log_vendor_invoice_count": (
        "vendor sends more invoices than usual",
        "vendor has few invoices on record",
    ),
    "vendor_mean_log_total": (
        "total small for what this vendor usually bills",
        "total large for what this vendor usually bills",
    ),
    "vendor_std_log_total": ("vendor's totals vary a lot", "vendor's totals hardly vary"),
}


def _detail(feature: FeatureError) -> str:
    """The number behind a phrase, where one helps (``" (z = +3.2)"``)."""
    value = feature.value
    if feature.name == "total_zscore":
        return f" (z = {value:+.1f})"
    if feature.name == "days_to_due" and value:
        return f" ({_days(value)})"
    if feature.name == "tax_ratio":
        return f" ({value:.0%} of the subtotal)"
    if feature.name == "line_count":
        return f" ({value:.0f} lines)"
    if feature.name == "log_days_since_last" and value:
        return f" ({_days(math.expm1(value))} since the previous one)"
    return ""


def _days(value: float) -> str:
    n = round(value)
    return f"{n} day" if n == 1 else f"{n} days"


def describe_feature(feature: FeatureError) -> str:
    """One feature's deviation in plain language: ``"total unusually high for this vendor
    (z = +3.2)"``."""
    phrases = FEATURE_PHRASES.get(feature.name)
    if phrases is None:  # a PCA text component
        return "wording unlike similar invoices"
    # how far off the model's expectation, but for the z-score which way from the vendor's
    # history: "lower than usual (z = +0.2)" would read as a contradiction
    up = feature.value >= 0 if feature.name == "total_zscore" else feature.above
    return phrases[0 if up else 1] + _detail(feature)


class Reason(BaseModel):
    """One line of why an invoice scored as it did (stored in ``anomaly_scores.reasons``)."""

    source: Literal["rule", "autoencoder", "summary"]
    code: str
    """The rule, the feature, or ``nothing_unusual``."""
    text: str
    share: float | None = None
    """The feature's share of the reconstruction error (autoencoder reasons)."""
    related_ids: list[str] = Field(default_factory=list)


def explain(verdict: AnomalyScore, top: int = TOP_FEATURES) -> list[Reason]:
    """Why ``verdict`` is what it is: every rule hit's own reason; if the autoencoder
    flagged it, its ``top`` worst-reconstructed features in plain language; if nothing
    flagged it, one summary line. Never empty."""
    reasons = [
        Reason(source="rule", code=hit.rule, text=hit.reason, related_ids=hit.related_ids)
        for hit in verdict.rule_hits
    ]
    if verdict.ae_flagged:
        total = sum(f.error for f in verdict.feature_errors) or 1.0
        reasons += [
            Reason(
                source="autoencoder",
                code=f.name,
                text=describe_feature(f),
                share=round(f.error / total, 3),
            )
            for f in verdict.feature_errors[:top]
        ]
    if not reasons:
        if verdict.ae_error is None:
            text = "nothing unusual: no rule fired (no anomaly model loaded)"
        else:
            text = (
                f"nothing unusual: no rule fired, and the reconstruction error "
                f"{verdict.ae_error:.3f} is under the {verdict.ae_threshold:.3f} threshold"
            )
        reasons.append(Reason(source="summary", code="nothing_unusual", text=text))
    return reasons


# --------------------------------------------------------------------------- scorer


class AnomalyScorer:
    """Scores invoices with the rules and, if it has one, the autoencoder."""

    def __init__(self, model: ReconstructionModel | None = None) -> None:
        self.model = model

    @property
    def model_version(self) -> str:
        return self.model.version if self.model is not None else RULES_ONLY

    @classmethod
    def from_config(cls, config: AnomalyConfig | None = None) -> AnomalyScorer:
        """The configured model version; rules only if it is missing or will not load."""
        config = config or get_config().anomaly
        path = find_model(repo_path(config.models_dir), config.version)
        if path is None:
            logger.warning(
                "no anomaly model %r in %s: scoring with the rules only",
                config.version,
                config.models_dir,
            )
            return cls(None)
        try:
            model = Autoencoder.load(path, combined=config.use_combined_threshold)
        except ImportError as exc:
            logger.warning("cannot load %s (%s): scoring with the rules only", path, exc)
            return cls(None)
        logger.info("anomaly model %s (threshold %.4f)", model.version, model.threshold)
        return cls(model)

    def autoencoder(
        self, ref: InvoiceRef, history: VendorHistory
    ) -> tuple[float | None, float, tuple[FeatureError, ...]]:
        """``(error, threshold, per-feature errors)``; ``(None, 0.0, ())`` without a model."""
        if self.model is None:
            return None, 0.0, ()
        values = invoice_features(ref.invoice, history)
        x_raw = np.array([feature_vector(values, self.model.features)], dtype=np.float64)
        x, reconstructed = self.model.reconstruct(x_raw)
        diff = x[0] - reconstructed[0]
        squared = diff**2
        per_feature = sorted(
            (
                FeatureError(name, float(x_raw[0, i]), float(squared[i]), bool(diff[i] > 0))
                for i, name in enumerate(self.model.features)
            ),
            key=lambda f: -f.error,
        )
        return float(squared.mean()), self.model.threshold, tuple(per_feature)

    def score(
        self, ref: InvoiceRef, context: RuleContext, history: VendorHistory | None = None
    ) -> AnomalyScore:
        """One invoice: every rule against ``context``, the autoencoder on its features
        with its vendor's ``history`` (what was known before it), combined."""
        hits = check_invoice(ref, context)
        error, threshold, per_feature = self.autoencoder(ref, history or VendorHistory())
        return combine(hits, error, threshold, per_feature)


@lru_cache(maxsize=1)
def get_scorer() -> AnomalyScorer:
    """The process's scorer: the model is loaded on first use, then kept."""
    return AnomalyScorer.from_config()


# --------------------------------------------------------------------------- Postgres


@dataclass(frozen=True)
class StoredContext:
    """What the scorer needs about one stored invoice, read from Postgres."""

    ref: InvoiceRef
    rules: RuleContext
    history: VendorHistory


def load_context(session: Session, invoice_id: str) -> StoredContext | None:
    """The invoice and what was known when it arrived: the same vendor's invoices stored
    before it (``created_at``, then id), the PO it cites less what those invoices already
    billed against it, the PO's contract (else the vendor's contract in force on the
    invoice date), the vendor master (vendors not first seen on a document) and the
    vendor's history for the features. ``None`` if there is no such invoice."""
    from sqlalchemy import and_, func, or_, select, tuple_
    from sqlalchemy.orm import selectinload

    from ledgerlens.storage.neo4j.linker import NEW_VENDOR_PREFIX
    from ledgerlens.storage.postgres.models import ContractRow, InvoiceRow, VendorRow
    from ledgerlens.storage.postgres.repositories import (
        PurchaseOrderRepository,
        to_contract,
        to_invoice,
        to_purchase_order,
    )

    row = session.get(InvoiceRow, invoice_id)
    if row is None:
        return None
    invoice = to_invoice(row)
    ref = InvoiceRef(row.id, invoice, row.vendor_id)
    earlier = tuple_(InvoiceRow.created_at, InvoiceRow.id) < tuple_(row.created_at, row.id)
    same_vendor = (
        InvoiceRow.vendor_id == row.vendor_id
        if row.vendor_id is not None
        else and_(InvoiceRow.vendor_id.is_(None), InvoiceRow.vendor_name == row.vendor_name)
    )
    stored = [
        InvoiceRef(other.id, to_invoice(other), other.vendor_id)
        for other in session.scalars(
            select(InvoiceRow)
            .where(same_vendor, earlier)
            .options(selectinload(InvoiceRow.line_items))
        )
    ]

    po = po_row = None
    billed = Decimal(0)
    if invoice.po_number:
        repo = PurchaseOrderRepository(session)
        rows = repo.find_by_number(invoice.po_number, vendor_id=row.vendor_id)
        rows = rows or repo.find_by_number(invoice.po_number)
        if rows:
            po_row = rows[0]
            po = to_purchase_order(po_row)
            billed = Decimal(
                session.scalar(
                    select(func.coalesce(func.sum(InvoiceRow.total), 0)).where(
                        InvoiceRow.po_number == invoice.po_number,
                        InvoiceRow.currency == po.currency,
                        earlier,
                    )
                )
                or 0
            )
    contract_row = po_row.contract if po_row is not None else None
    if contract_row is None and row.vendor_id is not None:
        contract_row = session.scalars(
            select(ContractRow)
            .where(
                ContractRow.vendor_id == row.vendor_id,
                ContractRow.start_date <= invoice.invoice_date,
                or_(ContractRow.end_date.is_(None), ContractRow.end_date >= invoice.invoice_date),
            )
            .order_by(ContractRow.start_date.desc(), ContractRow.id)
            .limit(1)
        ).first()

    vendors = session.scalars(
        select(VendorRow).where(VendorRow.id.not_like(f"{NEW_VENDOR_PREFIX}%"))
    )
    master = VendorMaster.from_vendors(
        {"vendor_id": v.id, "name": v.canonical_name, "tax_id": v.tax_id, "aliases": v.aliases}
        for v in vendors
    )
    if row.vendor_id is not None:
        history = load_vendor_history(
            session, row.vendor_id, invoice.invoice_date, exclude_id=row.id
        )
    else:
        history = VendorHistory.from_invoices(
            (s.invoice.invoice_date, s.invoice.total)
            for s in stored
            if s.invoice.invoice_date <= invoice.invoice_date
        )
    rules = RuleContext(
        stored=stored,
        purchase_order=po,
        already_billed=billed,
        contract=to_contract(contract_row) if contract_row is not None else None,
        master=master,
    )
    return StoredContext(ref, rules, history)


def score_stored(session: Session, invoice_id: str, scorer: AnomalyScorer) -> AnomalyScore | None:
    """Score a stored invoice and write its ``anomaly_scores`` row (score and reasons,
    under the scorer's model version). ``None`` if the invoice is not stored."""
    from ledgerlens.storage.postgres.repositories import AnomalyScoreRepository

    context = load_context(session, invoice_id)
    if context is None:
        return None
    verdict = scorer.score(context.ref, context.rules, context.history)
    AnomalyScoreRepository(session).upsert(
        invoice_id,
        scorer.model_version,
        verdict.score,
        [reason.model_dump(mode="json") for reason in verdict.reasons],
    )
    return verdict


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Score stored invoices: the ones given, and with --missing every invoice the current
    model version has not scored yet."""
    import argparse

    from ledgerlens.storage.postgres.db import get_session_factory, session_scope
    from ledgerlens.storage.postgres.repositories import AnomalyScoreRepository

    parser = argparse.ArgumentParser(
        prog="python -m ledgerlens.anomaly.scorer", description=main.__doc__
    )
    parser.add_argument("invoice_ids", nargs="*")
    parser.add_argument("--missing", action="store_true", help="every invoice not yet scored")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scorer = get_scorer()
    sessions = get_session_factory()
    ids = list(args.invoice_ids)
    if args.missing:
        with sessions() as session:
            ids += AnomalyScoreRepository(session).unscored(scorer.model_version)
    flagged = 0
    for n, invoice_id in enumerate(ids, 1):
        with session_scope(sessions) as session:
            verdict = score_stored(session, invoice_id, scorer)
        if verdict is None:
            print(f"{invoice_id}: no such invoice")
            continue
        flagged += verdict.flagged
        if args.invoice_ids:
            texts = "; ".join(reason.text for reason in verdict.reasons)
            print(
                f"{invoice_id}: {verdict.score:.3f} {'FLAG' if verdict.flagged else 'ok'} {texts}"
            )
        if n % 500 == 0:
            logger.info("scored %d of %d", n, len(ids))
    print(f"scored {len(ids)} invoices with {scorer.model_version}: {flagged} flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
