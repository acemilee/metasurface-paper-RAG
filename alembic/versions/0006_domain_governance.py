"""add domain governance and quarantine states"""

from alembic import op
import sqlalchemy as sa

revision = "0006_domain_governance"
down_revision = "0005_phase4_quality_and_recovery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE documentstatus ADD VALUE IF NOT EXISTS 'REVIEW_REQUIRED'")
    op.execute("ALTER TYPE documentstatus ADD VALUE IF NOT EXISTS 'QUARANTINED'")
    op.execute("ALTER TYPE jobstate ADD VALUE IF NOT EXISTS 'REVIEW_REQUIRED'")
    op.execute("ALTER TYPE jobstate ADD VALUE IF NOT EXISTS 'QUARANTINED'")
    op.add_column("documents", sa.Column("domain_status", sa.String(32), server_default="unclassified", nullable=False))
    op.add_column("documents", sa.Column("domain_score", sa.Float(), nullable=True))
    op.add_column("documents", sa.Column("domain_positive_score", sa.Float(), nullable=True))
    op.add_column("documents", sa.Column("domain_negative_score", sa.Float(), nullable=True))
    op.add_column("documents", sa.Column("domain_reasons_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("documents", sa.Column("domain_classifier_version", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("domain_checked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("domain_manual_override_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE documents SET domain_status = 'accepted' WHERE status = 'COMPLETED'")


def downgrade() -> None:
    for column in (
        "domain_manual_override_at",
        "domain_checked_at",
        "domain_classifier_version",
        "domain_reasons_json",
        "domain_negative_score",
        "domain_positive_score",
        "domain_score",
        "domain_status",
    ):
        op.drop_column("documents", column)
