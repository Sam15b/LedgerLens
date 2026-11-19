"""Combining rule hits with the autoencoder score."""

import numpy as np
import pytest

from ledgerlens.anomaly.rules import RuleHit
from ledgerlens.anomaly.scorer import RULE_BAND, ae_score, combine, combine_arrays

HIT = RuleHit(rule="exact_duplicate", invoice_id="S2", reason="already received as S1")


def test_ae_score_is_half_at_threshold() -> None:
    assert ae_score(0.2, 0.2) == 0.5
    assert ae_score(0.0, 0.2) == 0.0
    assert ae_score(None, 0.2) == 0.0
    assert 0.5 < ae_score(1.0, 0.2) < 1.0


def test_hard_hit_flags_and_outranks_the_autoencoder() -> None:
    ruled = combine([HIT], error=0.01, threshold=0.2)
    unusual = combine([], error=5.0, threshold=0.2)
    assert ruled.flagged and ruled.hard_hit
    assert unusual.flagged and not unusual.hard_hit
    assert ruled.score >= RULE_BAND > unusual.score


def test_without_hit_the_threshold_decides() -> None:
    assert not combine([], error=0.19, threshold=0.2).flagged
    assert combine([], error=0.2, threshold=0.2).flagged
    assert combine([], error=0.2, threshold=0.2).score == pytest.approx(0.25)
    assert not combine([], error=None, threshold=0.2).flagged


def test_combine_arrays_matches_combine() -> None:
    hard = np.array([True, False, False, True])
    errors = np.array([0.01, 0.5, 0.1, 0.0])
    scores, flags = combine_arrays(hard, errors, 0.2)
    for i in range(4):
        one = combine([HIT] if hard[i] else [], float(errors[i]), 0.2)
        assert scores[i] == pytest.approx(one.score)
        assert bool(flags[i]) == one.flagged
