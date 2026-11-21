"""A person's review of an anomaly flag, kept for retraining.

``anomaly_scores.review``: ``confirmed`` (a real problem) or ``false_positive``; ``NULL``
until someone reviews the flag. ``review_note`` and ``reviewed_at`` say why and when.
Re-scoring an invoice with the same model version keeps its review.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-21 07:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("anomaly_scores", sa.Column("review", sa.String(length=20), nullable=True))
    op.add_column("anomaly_scores", sa.Column("review_note", sa.Text(), nullable=True))
    op.add_column(
        "anomaly_scores", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint(
        op.f("ck_anomaly_scores_review"),
        "anomaly_scores",
        "review IN ('confirmed', 'false_positive')",
    )
    op.create_index(op.f("ix_anomaly_scores_review"), "anomaly_scores", ["review"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_anomaly_scores_review"), table_name="anomaly_scores")
    op.drop_constraint(op.f("ck_anomaly_scores_review"), "anomaly_scores", type_="check")
    op.drop_column("anomaly_scores", "reviewed_at")
    op.drop_column("anomaly_scores", "review_note")
    op.drop_column("anomaly_scores", "review")
