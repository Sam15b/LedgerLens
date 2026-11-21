"""End-to-end tests run against the compose services, like the integration tests: reuse
their throwaway-database fixtures (``pytest -m integration tests/e2e``)."""

from tests.integration.conftest import (  # noqa: F401  (fixtures)
    alembic,
    database_url,
    engine,
    migrated_engine,
    sessions,
)
