"""add normalized original filename search key"""

from alembic import op
import sqlalchemy as sa

from paper_rag.services.filename_search import normalize_filename_search_key


revision = "0020_filename_search_key"
down_revision = "0019_verified_formula_latex"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("filename_search_key", sa.Text(), nullable=True))
    connection = op.get_bind()
    rows = connection.execute(sa.text("SELECT id, original_filename FROM documents"))
    for row in rows.mappings():
        connection.execute(
            sa.text(
                "UPDATE documents SET filename_search_key = :search_key WHERE id = :document_id"
            ),
            {
                "search_key": normalize_filename_search_key(row["original_filename"]),
                "document_id": row["id"],
            },
        )
    op.alter_column("documents", "filename_search_key", nullable=False)


def downgrade() -> None:
    op.drop_column("documents", "filename_search_key")
