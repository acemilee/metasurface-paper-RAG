"""add independent formula index lifecycle"""

from alembic import op
import sqlalchemy as sa


revision = "0017_formula_index_status"
down_revision = "0016_formula_parser_default"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("formula_index_status", sa.String(32), nullable=False, server_default="pending"),
    )
    op.add_column(
        "documents",
        sa.Column("formula_parser_version", sa.String(64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("formula_index_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_formula_index_status", "documents", ["formula_index_status"])
    op.execute(
        """
        UPDATE documents
        SET formula_index_status = 'stale',
            formula_parser_version = (
                SELECT MIN(formulas.parser_version)
                FROM formulas
                WHERE formulas.document_id = documents.id
            ),
            formula_index_updated_at = CURRENT_TIMESTAMP
        WHERE EXISTS (
            SELECT 1 FROM formulas WHERE formulas.document_id = documents.id
        )
        """
    )


def downgrade() -> None:
    op.drop_index("ix_documents_formula_index_status", table_name="documents")
    op.drop_column("documents", "formula_index_updated_at")
    op.drop_column("documents", "formula_parser_version")
    op.drop_column("documents", "formula_index_status")
