"""add explainable document genre v2 fields"""

from alembic import op
import sqlalchemy as sa


revision = "0010_document_genre_v2"
down_revision = "0009_answer_actions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("genre_decision_source", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("genre_scores_json", sa.Text(), server_default="{}", nullable=False))
    op.add_column("documents", sa.Column("genre_evidence_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("documents", sa.Column("genre_conflicts_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("documents", sa.Column("genre_manually_overridden", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.add_column("documents", sa.Column("genre_original_prediction", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "genre_original_prediction")
    op.drop_column("documents", "genre_manually_overridden")
    op.drop_column("documents", "genre_conflicts_json")
    op.drop_column("documents", "genre_evidence_json")
    op.drop_column("documents", "genre_scores_json")
    op.drop_column("documents", "genre_decision_source")
