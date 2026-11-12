"""Live provider smoke test: the same prompts through every model in configs/models.yaml.

Makes real, billed API calls (a fraction of a cent per model), so run it by hand with real
keys in .env / AWS credentials, never in CI. For each model it

1. extracts a short invoice into a Pydantic schema (structured output), scored field by
   field against the known answer, and
2. streams a one-sentence answer, timing the first chunk.

    python scripts/llm_smoke.py                                  # every model in an alias
    python scripts/llm_smoke.py --models openai/gpt-4o-mini ollama/llama3:8b
    python scripts/llm_smoke.py --report eval/reports/2026-11-11/providers.md

Every call is also appended to the cost ledger (``cost_log`` in models.yaml).
Exit codes: 0 every model answered, 1 at least one model failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from ledgerlens.core.config import ModelsConfig, get_config
from ledgerlens.llm.router import LLMClient
from ledgerlens.llm.types import LLMError, Message

SYSTEM = "You extract data from accounts-payable documents. Copy values exactly as written."

INVOICE_TEXT = """\
INVOICE INV-2026-0417
Acme Supplies Ltd, 12 Harbour Road, Leeds LS1 4AB
Bill to: Northwind Traders
Invoice date: 2 October 2026        Due date: 1 November 2026 (Net 30)
Your PO number: PO-88231

  2 x Pallet wrap 500m           @ 45.00     90.00
 10 x Shipping labels (box/500)  @ 32.50    325.00

Subtotal                                    415.00
VAT 20%                                      83.00
Total due (GBP)                             498.00
"""

STREAM_PROMPT = "In one sentence, what does 'Net 30' mean on an invoice?"


class SmokeInvoice(BaseModel):
    """Small stand-in for the Day 9 Invoice schema."""

    invoice_number: str
    vendor_name: str
    po_number: str | None
    invoice_date: date
    due_date: date | None
    currency: str
    subtotal: Decimal
    tax: Decimal
    total: Decimal


EXPECTED = SmokeInvoice(
    invoice_number="INV-2026-0417",
    vendor_name="Acme Supplies Ltd",
    po_number="PO-88231",
    invoice_date=date(2026, 10, 2),
    due_date=date(2026, 11, 1),
    currency="GBP",
    subtotal=Decimal("415.00"),
    tax=Decimal("83.00"),
    total=Decimal("498.00"),
)


@dataclass
class SmokeResult:
    model_ref: str
    fields_correct: int | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    first_chunk_ms: float | None = None
    stream_text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def models_in_aliases(config: ModelsConfig) -> list[str]:
    """Every ``provider/model`` named in an alias, in first-mention order."""
    refs: dict[str, None] = {}
    for alias in config.aliases:
        refs.update(dict.fromkeys(config.resolve(alias)))
    return list(refs)


def score(invoice: SmokeInvoice) -> int:
    """Number of fields equal to the known answer (vendor compared case-insensitively)."""
    correct = 0
    for name in SmokeInvoice.model_fields:
        got, want = getattr(invoice, name), getattr(EXPECTED, name)
        if isinstance(want, str) and isinstance(got, str):
            correct += got.strip().casefold() == want.casefold()
        else:
            correct += got == want
    return correct


def run_model(client: LLMClient, ref: str) -> SmokeResult:
    result = SmokeResult(ref)
    messages = [Message.system(SYSTEM), Message.user(f"Extract this invoice:\n\n{INVOICE_TEXT}")]
    try:
        before = len(client.ledger.records)
        invoice = client.structured(messages, SmokeInvoice, model=ref, max_tokens=1024)
        calls = client.ledger.records[before:]  # one call, or two if it needed a repair
        result.fields_correct = score(invoice)
        result.latency_ms = sum(c.latency_ms for c in calls)
        result.tokens_in = sum(c.tokens_in for c in calls)
        result.tokens_out = sum(c.tokens_out for c in calls)
        costs = [c.cost_usd for c in calls]
        result.cost_usd = None if None in costs else sum(c or 0.0 for c in costs)

        start = time.perf_counter()
        chunks: list[str] = []
        for chunk in client.stream([Message.user(STREAM_PROMPT)], model=ref, max_tokens=200):
            if not chunks:
                result.first_chunk_ms = round((time.perf_counter() - start) * 1000, 1)
            chunks.append(chunk)
        result.stream_text = "".join(chunks).strip()
    except (LLMError, KeyError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def render_markdown(results: Sequence[SmokeResult], *, when: datetime) -> str:
    fields = len(SmokeInvoice.model_fields)
    lines = [
        f"# Provider comparison — {when:%Y-%m-%d}",
        "",
        f"Generated by `scripts/llm_smoke.py` at {when:%Y-%m-%d %H:%M} UTC. Each model got the "
        f"same invoice to extract into a {fields}-field schema (structured output, "
        "temperature 0) and one streamed one-sentence question. Prices from "
        "`configs/models.yaml`.",
        "",
        "| Model | Extraction fields correct | Extraction latency (ms) | Tokens in / out "
        "| Cost (USD) | Stream first chunk (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if not r.ok:
            lines.append(f"| `{r.model_ref}` | failed | | | | |")
            continue
        cost = "unpriced" if r.cost_usd is None else f"{r.cost_usd:.6f}"
        lines.append(
            f"| `{r.model_ref}` | {r.fields_correct}/{fields} | {r.latency_ms:.0f} "
            f"| {r.tokens_in} / {r.tokens_out} | {cost} | {r.first_chunk_ms:.0f} |"
        )
    failures = [r for r in results if not r.ok]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r.model_ref}`: {r.error}" for r in failures]
    lines += ["", "## Streamed answers", ""]
    lines += [f"- `{r.model_ref}`: {r.stream_text}" for r in results if r.ok]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="provider/model refs (default: all)")
    parser.add_argument("--report", type=Path, help="also write the markdown report here")
    args = parser.parse_args(argv)

    config = get_config().models
    client = LLMClient(config)
    refs = args.models or models_in_aliases(config)
    results = []
    for ref in refs:
        print(f"-> {ref} ...", file=sys.stderr, flush=True)
        results.append(run_model(client, ref))

    report = render_markdown(results, when=datetime.now(UTC))
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"report written to {args.report}", file=sys.stderr)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
