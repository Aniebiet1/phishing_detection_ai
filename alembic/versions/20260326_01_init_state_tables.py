"""Create user/session/prediction tables.

Revision ID: 20260326_01
Revises:
Create Date: 2026-03-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260326_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def ensure_index(table_name: str, index_name: str, cols: list[str], unique: bool = False) -> None:
        existing = {idx["name"] for idx in inspector.get_indexes(table_name)}
        if index_name not in existing:
            op.create_index(index_name, table_name, cols, unique=unique)

    if not inspector.has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("email", sa.String(length=254), nullable=False),
            sa.Column("salt", sa.String(length=64), nullable=False),
            sa.Column("password_hash", sa.String(length=128), nullable=False),
            sa.Column("joined_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_prediction_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
    else:
        ensure_index("users", "ix_users_email", ["email"], unique=True)

    if not inspector.has_table("sessions"):
        op.create_table(
            "sessions",
            sa.Column("token", sa.String(length=128), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)
        op.create_index("ix_sessions_token", "sessions", ["token"], unique=False)
    else:
        ensure_index("sessions", "ix_sessions_user_id", ["user_id"], unique=False)
        ensure_index("sessions", "ix_sessions_token", ["token"], unique=False)

    if not inspector.has_table("predictions"):
        op.create_table(
            "predictions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("preview", sa.Text(), nullable=False),
            sa.Column("label", sa.String(length=32), nullable=False),
            sa.Column("score", sa.Float(), nullable=False),
            sa.Column("source", sa.String(length=64), nullable=False),
            sa.Column("input_length", sa.Integer(), nullable=False),
        )
        op.create_index("ix_predictions_user_id", "predictions", ["user_id"], unique=False)
        op.create_index("ix_predictions_timestamp", "predictions", ["timestamp"], unique=False)
        op.create_index("ix_predictions_user_timestamp", "predictions", ["user_id", "timestamp"], unique=False)
    else:
        ensure_index("predictions", "ix_predictions_user_id", ["user_id"], unique=False)
        ensure_index("predictions", "ix_predictions_timestamp", ["timestamp"], unique=False)
        ensure_index("predictions", "ix_predictions_user_timestamp", ["user_id", "timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_predictions_user_timestamp", table_name="predictions")
    op.drop_index("ix_predictions_timestamp", table_name="predictions")
    op.drop_index("ix_predictions_user_id", table_name="predictions")
    op.drop_table("predictions")

    op.drop_index("ix_sessions_token", table_name="sessions")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
