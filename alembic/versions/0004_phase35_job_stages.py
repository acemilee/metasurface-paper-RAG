"""add phase 3.5 ingestion stages"""

from alembic import op

revision = "0004_phase35_job_stages"
down_revision = "0003_phase3_answer_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE jobstate ADD VALUE IF NOT EXISTS 'CHUNKING'")
    op.execute("ALTER TYPE jobstate ADD VALUE IF NOT EXISTS 'EMBEDDING'")
    op.execute("ALTER TYPE jobstate ADD VALUE IF NOT EXISTS 'INDEXING'")


def downgrade() -> None:
    # PostgreSQL enum labels cannot be removed safely without rebuilding the type.
    pass
