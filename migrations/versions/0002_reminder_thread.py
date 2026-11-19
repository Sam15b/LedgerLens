"""Reminders remember the agent run that drafted them.

``reminders.thread_id``: the LangGraph thread waiting for a decision on the draft, so
``POST /reminders/{id}/approve|reject`` can resume that run. ``NULL`` for drafts made
outside a run (the MCP tool called directly).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-20 03:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("reminders", sa.Column("thread_id", sa.String(length=100), nullable=True))
    op.create_index(op.f("ix_reminders_thread_id"), "reminders", ["thread_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_reminders_thread_id"), table_name="reminders")
    op.drop_column("reminders", "thread_id")
