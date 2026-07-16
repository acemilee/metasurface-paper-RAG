"""add explicit answer action and unsupported parts"""

from alembic import op
import sqlalchemy as sa

revision = "0009_answer_actions"
down_revision = "0008_semantic_claim_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("answer_audits", sa.Column("action", sa.String(length=32), server_default="refuse", nullable=False))
    op.add_column("answer_audits", sa.Column("unsupported_parts_json", sa.Text(), server_default="[]", nullable=False))


def downgrade() -> None:
    op.drop_column("answer_audits", "unsupported_parts_json")
    op.drop_column("answer_audits", "action")
