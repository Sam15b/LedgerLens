"""The Cypher templates: parameters bind strictly, and no template can write."""

import re

import pytest

from ledgerlens.storage.neo4j.queries import INVOICE_NEIGHBORHOOD, TEMPLATES, UNPAID_WITH_PO

WRITE_CLAUSE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b", re.I)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_templates_are_read_only(name: str) -> None:
    assert WRITE_CLAUSE.search(TEMPLATES[name].cypher) is None


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_declared_parameter_is_used_and_every_used_one_declared(name: str) -> None:
    template = TEMPLATES[name]
    used = set(re.findall(r"\$(\w+)", template.cypher))
    assert used == set(template.params)


def test_bind_fills_defaults_and_overrides() -> None:
    params = UNPAID_WITH_PO.bind(vendor_ids=["V001"], limit=5)
    assert params == {
        "vendor_ids": ["V001"],
        "date_from": None,
        "date_to": None,
        "tolerance": 0.01,
        "limit": 5,
    }


def test_bind_rejects_unknown_and_missing_parameters() -> None:
    with pytest.raises(ValueError, match="takes no parameter"):
        INVOICE_NEIGHBORHOOD.bind(seeds=["x"], vendor="Acme")
    with pytest.raises(ValueError, match="needs seeds"):
        INVOICE_NEIGHBORHOOD.bind()
