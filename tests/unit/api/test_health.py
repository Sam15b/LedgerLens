from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api.health_checks import HealthCheck
from ledgerlens.api.main import create_app
from ledgerlens.api.routers.health import get_health_checks


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_health_checks] = lambda: [HealthCheck("store", lambda: None)]
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
