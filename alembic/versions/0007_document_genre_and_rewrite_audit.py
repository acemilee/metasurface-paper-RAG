"""add document genre and query rewrite audit fields"""

from alembic import op
import sqlalchemy as sa

revision = "0007_genre_rewrite_audit"
down_revision = "0006_domain_governance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("document_genre", sa.String(32), server_default="unclassified", nullable=False))
    op.add_column("documents", sa.Column("genre_score", sa.Float(), nullable=True))
    op.add_column("documents", sa.Column("genre_classifier_version", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("genre_checked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("answer_audits", sa.Column("selected_document_ids_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("answer_audits", sa.Column("query_plan_json", sa.Text(), server_default="{}", nullable=False))
    op.add_column("answer_audits", sa.Column("entity_links_json", sa.Text(), server_default="[]", nullable=False))
    op.add_column("answer_audits", sa.Column("rewrite_source", sa.String(64), nullable=True))
    op.add_column("answer_audits", sa.Column("rewrite_error_json", sa.Text(), server_default="{}", nullable=False))
    op.add_column("answer_audits", sa.Column("document_genres_json", sa.Text(), server_default="{}", nullable=False))


def downgrade() -> None:
    for column in (
        "document_genres_json", "rewrite_error_json", "rewrite_source",
        "entity_links_json", "query_plan_json", "selected_document_ids_json",
    ):
        op.drop_column("answer_audits", column)
    for column in ("genre_checked_at", "genre_classifier_version", "genre_score", "document_genre"):
        op.drop_column("documents", column)
