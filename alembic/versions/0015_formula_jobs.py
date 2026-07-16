"""add persistent formula backfill jobs"""

from alembic import op
import sqlalchemy as sa


revision = "0015_formula_jobs"
down_revision = "0014_formula_layout_v3"
branch_labels = None
depends_on = None


formula_backfill_job_state = sa.Enum(
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "NEEDS_REVIEW",
    "FAILED",
    "CANCELLED",
    name="formula_backfill_job_state",
)


def upgrade() -> None:
    op.create_table(
        "formula_backfill_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("state", formula_backfill_job_state, nullable=False),
        sa.Column("page_numbers_json", sa.Text(), nullable=False),
        sa.Column("source_parser_versions_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("target_parser_version", sa.String(64), nullable=False),
        sa.Column("apply_safe", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("inventory_signature", sa.String(64), nullable=True),
        sa.Column("worker_id", sa.String(128), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_formula_backfill_jobs_document_id", "formula_backfill_jobs", ["document_id"])
    op.create_index("ix_formula_backfill_jobs_state", "formula_backfill_jobs", ["state"])


def downgrade() -> None:
    op.drop_index("ix_formula_backfill_jobs_state", table_name="formula_backfill_jobs")
    op.drop_index("ix_formula_backfill_jobs_document_id", table_name="formula_backfill_jobs")
    op.drop_table("formula_backfill_jobs")
    formula_backfill_job_state.drop(op.get_bind(), checkfirst=True)
