from ledgerlens.storage.neo4j.driver import apply_schema, load_schema, split_cypher


def test_split_cypher_drops_comments_and_blank_statements() -> None:
    script = """
    // header comment
    CREATE CONSTRAINT a IF NOT EXISTS FOR (n:A) REQUIRE n.id IS UNIQUE;

    // another
    CREATE INDEX b IF NOT EXISTS
      FOR (n:B) ON (n.x);
    ;
    """

    statements = split_cypher(script)

    assert len(statements) == 2
    assert statements[0].startswith("CREATE CONSTRAINT a")
    assert "FOR (n:B) ON (n.x)" in statements[1]


def test_packaged_schema_is_idempotent_and_covers_core_labels() -> None:
    statements = load_schema()

    assert statements
    assert all("IF NOT EXISTS" in s for s in statements)
    joined = "\n".join(statements)
    for label in ("Vendor", "Invoice", "PurchaseOrder", "Payment", "Contract"):
        assert f":{label})" in joined
    assert "document_id" in joined


class RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def execute_query(self, query: str, /, *, database_: str | None = None) -> object:
        self.calls.append((query, database_))
        return None


def test_apply_schema_runs_every_statement() -> None:
    driver = RecordingDriver()

    count = apply_schema(driver, ["RETURN 1", "RETURN 2"], database="neo4j")

    assert count == 2
    assert driver.calls == [("RETURN 1", "neo4j"), ("RETURN 2", "neo4j")]


def test_apply_schema_defaults_to_packaged_schema() -> None:
    driver = RecordingDriver()

    assert apply_schema(driver) == len(load_schema())
