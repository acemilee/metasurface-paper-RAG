"""add phase 4 extraction quality and worker recovery"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_phase4_quality_and_recovery"
down_revision = "0004_phase35_job_stages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("extraction_method", sa.String(32), server_default="digital_text", nullable=False))
    op.add_column("pages", sa.Column("quality_score", sa.Float(), server_default="1", nullable=False))
    op.add_column("pages", sa.Column("ocr_confidence", sa.Float(), nullable=True))
    op.add_column("text_blocks", sa.Column("source", sa.String(32), server_default="digital_text", nullable=False))
    op.add_column("text_blocks", sa.Column("confidence", sa.Float(), server_default="1", nullable=False))
    op.add_column("chunks", sa.Column("quality_score", sa.Float(), server_default="1", nullable=False))
    op.add_column("chunks", sa.Column("has_low_confidence_ocr", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.add_column("ingestion_jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False))
    op.add_column("ingestion_jobs", sa.Column("stage_durations_json", sa.Text(), server_default="{}", nullable=False))
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(128), primary_key=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(32), server_default="idle", nullable=False),
    )


def downgrade() -> None:
    op.drop_table("worker_heartbeats")
    for column in ("stage_durations_json", "attempt_count", "heartbeat_at"):
        op.drop_column("ingestion_jobs", column)
    for column in ("has_low_confidence_ocr", "quality_score"):
        op.drop_column("chunks", column)
    for column in ("confidence", "source"):
        op.drop_column("text_blocks", column)
    for column in ("ocr_confidence", "quality_score", "extraction_method"):
        op.drop_column("pages", column)
