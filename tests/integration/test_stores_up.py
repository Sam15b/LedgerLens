"""Connectivity tests against the real stores from docker-compose.

    docker compose up -d --wait postgres neo4j redis pinecone-local
    PINECONE_HOST=http://localhost:5080 pytest -m integration

(Set DATABASE_URL / POSTGRES_PORT too if the compose Postgres is not on 5432.)
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api import deps
from ledgerlens.api.health_checks import HealthCheck, default_checks
from ledgerlens.api.main import create_app
from ledgerlens.core.config import get_config, get_settings
from ledgerlens.core.ids import chunk_vector_id, new_document_id
from ledgerlens.storage.neo4j.driver import apply_schema
from ledgerlens.storage.pinecone_store import create_client, ensure_index

pytestmark = pytest.mark.integration

REQUIRED = [c for c in default_checks() if c.required]


@pytest.mark.parametrize("check", REQUIRED, ids=[c.name for c in REQUIRED])
def test_store_is_reachable(check: HealthCheck) -> None:
    check.check()  # raises if the store is unreachable


def test_ready_endpoint_with_real_stores() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/ready")

    body = response.json()
    assert response.status_code == 200, body
    assert all(body["dependencies"][c.name]["status"] == "ok" for c in REQUIRED)


def test_neo4j_schema_applies_idempotently() -> None:
    driver = deps.get_neo4j_driver()

    first = apply_schema(driver)
    second = apply_schema(driver)
    records, _, _ = driver.execute_query("SHOW CONSTRAINTS YIELD name RETURN name")

    assert first == second > 0
    names = {r["name"] for r in records}
    assert {"vendor_id", "invoice_id", "po_id", "payment_id", "contract_id"} <= names


@pytest.mark.skipif(
    not get_settings().pinecone_host,
    reason="creates and deletes an index: only run against Pinecone Local (PINECONE_HOST)",
)
def test_pinecone_index_lifecycle_and_hybrid_query() -> None:
    settings, shape = get_settings(), get_config().pinecone
    admin = deps.get_pinecone_admin()
    name = f"it-{uuid.uuid4().hex[:8]}"
    kwargs = {"cloud": settings.pinecone_cloud, "region": settings.pinecone_region}
    try:
        outcome, info = ensure_index(
            admin, name, dimension=shape.dimension, metric=shape.metric, **kwargs
        )
        again, _ = ensure_index(
            admin, name, dimension=shape.dimension, metric=shape.metric, **kwargs
        )
        assert (outcome, again) == ("created", "exists")
        assert info.host

        index = create_client(settings.pinecone_api_key, settings.pinecone_host).Index(
            host=info.host
        )
        vector_id = chunk_vector_id(new_document_id(), 0)
        index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "values": [0.1] * shape.dimension,
                    "sparse_values": {"indices": [3, 7], "values": [0.5, 0.2]},
                    "metadata": {"doc_type": "invoice"},
                }
            ],
            namespace="it",
        )
        result = index.query(
            vector=[0.1] * shape.dimension,
            sparse_vector={"indices": [3], "values": [1.0]},
            top_k=1,
            namespace="it",
            include_metadata=True,
        )
        assert result.matches[0].id == vector_id
        assert result.matches[0].metadata == {"doc_type": "invoice"}
    finally:
        admin.delete(name)
