import pytest

from eval.ocr_cer import CERResult, cer, cer_counts, corpus_cer, levenshtein, normalize_text


@pytest.mark.parametrize(
    "a,b,distance",
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("TOTAL 9.50", "TOTAL 9.50", 0),
        ("9.50", "9,50", 1),
    ],
)
def test_levenshtein(a: str, b: str, distance: int) -> None:
    assert levenshtein(a, b) == levenshtein(b, a) == distance


def test_normalization_ignores_whitespace_and_case() -> None:
    assert normalize_text("  TOTAL\n\t MYR  9.50 ") == "total myr 9.50"
    assert cer("TOTAL MYR\n9.50", "total  myr 9.50") == 0.0
    assert cer("TOTAL", "total", normalize=False) == 1.0


def test_cer_is_errors_over_reference_length() -> None:
    assert cer_counts("Kopi O 4.40", "Kopi 0 4.4") == CERResult(errors=2, reference_chars=11)
    assert cer("Kopi O 4.40", "Kopi 0 4.4") == pytest.approx(2 / 11)


def test_empty_reference() -> None:
    assert cer("", "") == 0.0
    assert cer("", "noise") == float("inf")


def test_corpus_cer_weights_by_reference_length() -> None:
    pairs = [("a" * 90, "a" * 90), ("b" * 10, "c" * 10)]

    assert corpus_cer(pairs) == pytest.approx(10 / 100)  # mean of per-doc CERs would be 0.5
