import pytest

from ledgerlens.storage.neo4j.linker import (
    VendorCandidate,
    VendorMatch,
    new_vendor_id,
    normalize_tax_id,
    resolve_vendor,
)

VENDORS = [
    VendorCandidate("V001", "Acme Supplies Ltd", "GB123456789", ("ACME SUPPLIES",)),
    VendorCandidate("V002", "Brightline Packaging Ltd", "GB284615093"),
    VendorCandidate("V003", "Northwind Logistics GmbH", None, ("NW Logistics",)),
]


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("gb 123-456 789", "GB123456789"),
        ("DE.811/907.980", "DE811907980"),
        (" ", None),
        (None, None),
    ],
)
def test_normalize_tax_id(raw: str | None, clean: str | None) -> None:
    assert normalize_tax_id(raw) == clean


def test_tax_id_wins_over_a_different_name() -> None:
    match = resolve_vendor("Totally Different Trading", "gb 284 615 093", VENDORS)

    assert match == VendorMatch("V002", "tax_id", 100.0)


@pytest.mark.parametrize(
    ("printed", "vendor_id"),
    [
        ("ACME Supplies, Limited", "V001"),  # legal form and punctuation ignored
        ("acme supplies", "V001"),  # an alias
        ("NW LOGISTICS", "V003"),
    ],
)
def test_exact_canonical_name_or_alias(printed: str, vendor_id: str) -> None:
    assert resolve_vendor(printed, None, VENDORS) == VendorMatch(vendor_id, "name", 100.0)


def test_unknown_tax_id_falls_back_to_the_name() -> None:
    match = resolve_vendor("Acme Supplies Ltd", "FR00000000000", VENDORS)

    assert match is not None and (match.vendor_id, match.method) == ("V001", "name")


@pytest.mark.parametrize("printed", ["ACME SUPPLES LTD", "Supplies Acme", "Brightline Packagng"])
def test_fuzzy_match_survives_ocr_slips_and_word_order(printed: str) -> None:
    match = resolve_vendor(printed, None, VENDORS)

    assert match is not None and match.method in {"fuzzy", "name"}
    assert match.vendor_id in {"V001", "V002"}
    assert match.score >= 88


def test_threshold_controls_fuzzy_matches() -> None:
    assert resolve_vendor("Acme Supply Co", None, VENDORS) is None
    loose = resolve_vendor("Acme Supply Co", None, VENDORS, threshold=70)
    assert loose is not None and (loose.vendor_id, loose.method) == ("V001", "fuzzy")


@pytest.mark.parametrize("printed", ["", "  ", "Ltd."])
def test_names_with_nothing_to_compare_do_not_match(printed: str) -> None:
    assert resolve_vendor(printed, None, VENDORS) is None


def test_new_vendor_ids_are_stable_across_spellings() -> None:
    first = new_vendor_id("Kedai Kopi Sdn Bhd")

    assert first == new_vendor_id("KEDAI KOPI SDN. BHD.")
    assert first.startswith("v_kedai_kopi")
    assert first != new_vendor_id("Kedai Teh")
    assert len(new_vendor_id("x" * 300)) <= 64  # fits the Postgres vendors.id column
