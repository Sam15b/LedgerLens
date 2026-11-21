# Layer 4 · Agents (LangGraph) + Tools (MCP)

**Code:** `src/ledgerlens/agents/`, `src/ledgerlens/mcp_server/` · **Phase:** 5

## Purpose
A **router agent** understands the user's request and hands off to specialist agents
(QA, extractor, reconciler, reminder writer). All data access happens through tools
exposed by an **MCP server**, so the same tools work from LangGraph, Claude Desktop,
or any MCP client.

## Agent graph

```mermaid
stateDiagram-v2
    [*] --> Router
    Router --> QA: intent = question
    Router --> Extractor: intent = extract
    Router --> Reconciler: intent = reconcile
    Router --> ReminderWriter: intent = remind
    Router --> [*]: intent = chitchat / out_of_scope
    QA --> Respond
    Extractor --> Reconciler: auto-reconcile new invoice
    Extractor --> Respond
    Reconciler --> ReminderWriter: overdue & matched
    Reconciler --> Respond
    ReminderWriter --> HumanApproval
    HumanApproval --> Respond
    Respond --> [*]
```

## Shared state (`state.py`)

```python
class LedgerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["question", "extract", "reconcile", "remind", "other"] | None
    document_ids: list[str]
    extracted: dict | None              # Invoice/PO record
    reconciliation: dict | None         # match result + diffs
    reminder_drafts: list[dict]
    citations: dict[str, str]
    step_count: int
    errors: list[str]
```

## Agents

| Node | Model alias | Tools used | Output |
|------|-------------|------------|--------|
| **Router** | `fast` (e.g. GPT-4o-mini / Gemini Flash / Llama 3) | none — structured output | `intent`, extracted entities |
| **QA** | `smart` | `search_documents`, `graph_query`, `run_analytics` | Answer + citations |
| **Extractor** | `extractor` (fine-tuned Llama 3) | `get_document_text`, `save_extraction` | Typed record |
| **Reconciler** | `smart` | `get_invoice`, `get_po`, `get_contract`, `get_payments`, `reconcile` | Match / mismatch report |
| **Reminder writer** | `smart` | `list_overdue`, `get_vendor_contact`, `draft_reminder` | Drafts pending approval |

### Reconciler logic (deterministic core, LLM explains)
The 3-way match is **code**, not LLM judgment — the LLM only narrates results.

| Check | Rule (configurable tolerance) |
|-------|-------------------------------|
| PO exists | invoice.po_number resolves in graph |
| Vendor match | invoice.vendor_id == po.vendor_id |
| Line qty | Σ invoiced qty ≤ PO qty |
| Unit price | \|invoice − contract rate\| ≤ 2 % |
| Total | \|invoice.total − Σ lines − tax\| ≤ 0.01 |
| Payment | Σ payments vs total → paid / partial / over / unpaid |
| Terms | due_date consistent with contract `payment_terms_days` |

### Reminder writer
- Tone by days overdue: 1–15 friendly · 16–45 firm · 46+ final notice.
- Must cite invoice number, amount, original due date, PO number.
- Uses LangGraph `interrupt()` → drafts stored as `pending_approval`; resumes on approve/reject.
- **Never sends** in v1.

### Guardrails
- `step_count` limit (e.g. 8) to prevent loops.
- Tool errors are returned to the agent as messages, not raised.
- Every answer must have ≥1 citation or explicitly say evidence wasn't found.

## MCP server

Built with the Python MCP SDK (`FastMCP`). Runs over **stdio** (local clients) or
**streamable HTTP** (container in compose).

| Tool | Args | Returns |
|------|------|---------|
| `search_documents` | `query, doc_type?, vendor?, date_from?, date_to?, k=5` | chunks + document_ids |
| `graph_query` | `template, params` or `question` (Text2Cypher) | rows |
| `run_analytics` | `report` (aging / spend_by_vendor / unpaid) + params | table |
| `get_invoice` / `get_po` / `get_contract` | `id or number` | record |
| `get_payments` | `invoice_id` | payments |
| `list_overdue` | `min_days, vendor?` | invoices |
| `reconcile` | `invoice_id` | match report |
| `score_anomaly` | `invoice_id` | score + reasons |
| `draft_reminder` | `invoice_id, body, tone` | reminder id (status pending) |
| `get_vendor_contact` | `vendor` | name, email (may be `null`) |
| `get_document_text` | `document` (id or record number), `max_chars?` | text as read at ingestion + doc type |
| `save_extraction` | `document_id, record` | stored record id (+ anomaly verdict) |

```python
# mcp_server/tools/search.py (sketch)
@mcp.tool()
def search_documents(query: str, doc_type: str | None = None,
                     vendor: str | None = None, k: int = 5) -> list[dict]:
    """Semantic + keyword search over invoices, POs and contracts."""
    return retriever.retrieve(query, filters=QueryFilters(doc_type=doc_type, vendor=vendor), k=k).to_tool_output()
```

LangGraph loads the tools via `langchain-mcp-adapters` (`MultiServerMCPClient`), and each
agent node is bound only to the tools it needs.

### Connecting external MCP clients

The same 14 tools work from any MCP client. The server reads `.env` from the repo root.
It needs the stores up (`make up`), and for `search_documents` it also needs
`PINECONE_HOST=http://localhost:5080` when you use Pinecone Local. Without it, the
retriever warm-up logs a `PineconeNotConfiguredError` and every other tool still works.

**Over stdio** (the client starts the server):

```bash
python -m ledgerlens.mcp_server.server                  # what a stdio client runs
```

- **Claude Desktop**: add the server to `claude_desktop_config.json`
  (`%APPDATA%\Claude\` on Windows, `~/Library/Application Support/Claude/` on macOS).
  Use absolute paths, because Claude Desktop does not start in the repo:

  ```json
  {
    "mcpServers": {
      "ledgerlens": {
        "command": "E:/LedgerLens/.venv/Scripts/python.exe",
        "args": ["-m", "ledgerlens.mcp_server.server"],
        "env": { "PINECONE_HOST": "http://localhost:5080" }
      }
    }
  }
  ```

  Restart Claude Desktop. The tools then appear under *ledgerlens*.
- **Claude Code**: `claude mcp add ledgerlens -- <venv python> -m ledgerlens.mcp_server.server`.
- **MCP Inspector**: `make mcp-inspect`. It needs Node.js and runs
  `npx @modelcontextprotocol/inspector` on the stdio server. Open the URL it prints,
  then *Connect* → *Tools* → *List Tools*.

**Over streamable HTTP** (one long-running server for several clients; this is how the
agents connect):

```bash
make mcp                                                # http://localhost:8765/mcp
```

- **Claude Code**: `claude mcp add --transport http ledgerlens http://localhost:8765/mcp`.
- **MCP Inspector**: run `npx @modelcontextprotocol/inspector`, choose transport
  *Streamable HTTP* and enter the URL `http://localhost:8765/mcp`.
- **Python** (`mcp` SDK), the client this was checked with:

  ```python
  from mcp import ClientSession
  from mcp.client.streamable_http import streamablehttp_client

  async with streamablehttp_client("http://localhost:8765/mcp") as (read, write, _):
      async with ClientSession(read, write) as session:
          await session.initialize()
          print([t.name for t in (await session.list_tools()).tools])   # 14 tools
          await session.call_tool("list_overdue", {"min_days": 60, "limit": 5})
  ```

**Before you connect a client to real data:** most tools only read. `draft_reminder`
only stores a `pending_approval` draft, and nothing is ever sent. `save_extraction`
**overwrites** the stored record of a document and relinks it in the graph, so don't
let an external client call it on the dev ledger unless you mean to.

## API integration
- `POST /chat` → runs the graph, streams via **SSE**: `token`, `tool_call`, `tool_result`, `citation`, `interrupt`, `done` events.
- `POST /reminders/{id}/approve|reject` → resumes the interrupted graph (checkpointer: Postgres).

## Tests
- Router intent accuracy on a labeled set of ~100 utterances.
- Reconciler unit tests over synthetic cases (each mismatch type).
- Agent trajectory tests: expected tool-call sequence for each user story.

## Definition of done
- All user stories pass end-to-end via `/chat`.
- MCP server usable from MCP Inspector / an external MCP client.
- Traces show router → specialist hand-offs and tool calls.
