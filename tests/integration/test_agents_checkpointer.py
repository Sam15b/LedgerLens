"""The Postgres checkpointer: a run paused for reminder approval survives a restart
(a new saver and a new graph pick it up by thread id). Throwaway database.

    docker compose up -d --wait postgres
    pytest -m integration tests/integration/test_agents_checkpointer.py
"""

import asyncio
import sys
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.engine import URL

from ledgerlens.agents.graph import AgentNodes, build_graph, postgres_checkpointer
from ledgerlens.agents.state import LedgerState

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> Any:
    # psycopg's async mode cannot use Windows' default Proactor event loop
    if sys.platform == "win32":
        return ("asyncio", {"loop_factory": asyncio.SelectorEventLoop})
    return "asyncio"


def router(state: LedgerState) -> dict[str, Any]:
    return {"intent": "remind", "step_count": 1}


def writer(state: LedgerState) -> dict[str, Any]:
    draft = {"reminder_id": "r-1", "invoice_number": "INV-889", "tone": "firm", "body": "..."}
    return {"reminder_drafts": [draft], "step_count": 1}


async def test_paused_run_resumes_after_a_restart(database_url: URL) -> None:
    url = database_url.render_as_string(hide_password=False)
    thread = {"configurable": {"thread_id": "reminders-1"}}
    nodes = AgentNodes(router=router, reminder_writer=writer)

    async with postgres_checkpointer(url) as saver:
        first = await build_graph(nodes, checkpointer=saver).ainvoke(
            {"messages": [HumanMessage("Chase INV-889")]}, thread
        )
    assert first["__interrupt__"][0].value["reminders"][0]["reminder_id"] == "r-1"

    # "restart": a new connection and a new graph, nothing kept in memory
    async with postgres_checkpointer(url) as saver:
        graph = build_graph(nodes, checkpointer=saver)
        state = await graph.aget_state(thread)
        assert state.next == ("human_approval",)
        assert state.values["reminder_drafts"][0]["invoice_number"] == "INV-889"
        done = await graph.ainvoke(Command(resume={"r-1": "rejected"}), thread)
    assert done["reminder_decisions"] == {"r-1": "rejected"}
    assert done["messages"][-1].content.startswith("Rejected: INV-889.")
