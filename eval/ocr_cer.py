"""Character error rate (CER) for OCR output.

    CER = (substitutions + deletions + insertions) / characters in the reference

computed with Levenshtein edit distance. 0.0 is a perfect read; it can exceed 1.0 when
the OCR output is much longer than the reference.

By default both texts are normalized first (:func:`normalize_text`): whitespace runs
collapse to one space and case is ignored, so the score measures characters read, not
line breaks or capitalization, which the extractor does not depend on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str, *, casefold: bool = True) -> str:
    text = _WHITESPACE.sub(" ", text).strip()
    return text.casefold() if casefold else text


def levenshtein(a: str, b: str) -> int:
    """Edit distance between ``a`` and ``b`` (two-row dynamic programming)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # delete ca
                    current[j - 1] + 1,  # insert cb
                    previous[j - 1] + (ca != cb),  # substitute
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CERResult:
    errors: int
    reference_chars: int

    @property
    def cer(self) -> float:
        if self.reference_chars == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_chars


def cer_counts(reference: str, hypothesis: str, *, normalize: bool = True) -> CERResult:
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    return CERResult(levenshtein(reference, hypothesis), len(reference))


def cer(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """CER of ``hypothesis`` (OCR output) against ``reference`` (ground truth)."""
    return cer_counts(reference, hypothesis, normalize=normalize).cer


def corpus_cer(pairs: Iterable[tuple[str, str]], *, normalize: bool = True) -> float:
    """CER over many documents: total errors / total reference characters.

    Weighting by length means a long receipt counts more than a short one, unlike the
    mean of per-document CERs.
    """
    errors = chars = 0
    for reference, hypothesis in pairs:
        result = cer_counts(reference, hypothesis, normalize=normalize)
        errors += result.errors
        chars += result.reference_chars
    return CERResult(errors, chars).cer
