"""The dashboard's typed client is generated from ``web/openapi.json``: it must match the API."""

from pathlib import Path

import pytest

from scripts.export_openapi import DEFAULT_OUT, main, openapi_schema, render


def test_committed_schema_matches_the_api() -> None:
    committed = DEFAULT_OUT.read_text(encoding="utf-8")
    assert committed == render(openapi_schema()), (
        "web/openapi.json is stale: run python scripts/export_openapi.py && pnpm --dir web gen:api"
    )


def test_check_mode_reports_a_stale_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "openapi.json"
    assert main(["--out", str(out), "--check"]) == 1  # missing
    assert main(["--out", str(out)]) == 0
    assert main(["--out", str(out), "--check"]) == 0
    assert '"/health"' in out.read_text(encoding="utf-8")
    assert "is out of date" in capsys.readouterr().err
