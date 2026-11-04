"""Apply the Neo4j constraints and indexes from schema.cypher. Safe to run repeatedly.

Uses NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD from the environment or .env:

    python scripts/init_neo4j_schema.py

Exit codes: 0 schema applied, 1 Neo4j unreachable or authentication failed.
"""

from __future__ import annotations

import sys

from neo4j.exceptions import AuthError, ServiceUnavailable

from ledgerlens.core.config import get_settings
from ledgerlens.storage.neo4j.driver import apply_schema, create_driver


def main() -> int:
    settings = get_settings()
    driver = create_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        driver.verify_connectivity()
        count = apply_schema(driver)
        constraints, _, _ = driver.execute_query("SHOW CONSTRAINTS YIELD name RETURN name")
        indexes, _, _ = driver.execute_query(
            "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"
        )
    except (ServiceUnavailable, AuthError) as exc:
        print(f"error: cannot use Neo4j at {settings.neo4j_uri}: {exc}")
        return 1
    finally:
        driver.close()

    print(f"applied {count} schema statements to {settings.neo4j_uri}")
    print(f"constraints ({len(constraints)}): {', '.join(sorted(r['name'] for r in constraints))}")
    print(f"indexes ({len(indexes)}): {', '.join(sorted(r['name'] for r in indexes))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
