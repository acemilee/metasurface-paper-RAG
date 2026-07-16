"""add generation attempts and semantic claim audit trace"""

from alembic import op
import sqlalchemy as sa

revision = "0008_semantic_claim_audit"
down_revision = "0007_genre_rewrite_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("answer_audits", sa.Column("generation_attempts_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("answer_audits", sa.Column("semantic_audit_json", sa.Text(), server_default="[]", nullable=False))


def downgrade() -> None:
    op.drop_column("answer_audits", "semantic_audit_json")
    op.drop_column("answer_audits", "generation_attempts_json")
