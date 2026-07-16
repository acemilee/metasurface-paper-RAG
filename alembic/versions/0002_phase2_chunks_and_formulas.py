"""add phase 2 chunk and formula fields"""

from alembic import op
import sqlalchemy as sa

revision = "0002_phase2_chunks_and_formulas"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("section_path", sa.String(512), nullable=True))
    op.add_column("chunks", sa.Column("content_type", sa.String(64), nullable=True))
    op.add_column("chunks", sa.Column("formula_ids_json", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("chunk_index", sa.Integer(), nullable=True))
    op.execute("UPDATE chunks SET page_start = 1, page_end = 1, content_type = 'paragraph', formula_ids_json = '[]', chunk_index = 0 WHERE page_start IS NULL")
    op.alter_column("chunks", "page_start", nullable=False)
    op.alter_column("chunks", "page_end", nullable=False)
    op.alter_column("chunks", "content_type", nullable=False)
    op.alter_column("chunks", "formula_ids_json", nullable=False)
    op.alter_column("chunks", "chunk_index", nullable=False)
    op.add_column("formulas", sa.Column("raw_text", sa.Text(), nullable=True))
    op.add_column("formulas", sa.Column("context_before", sa.Text(), nullable=True))
    op.add_column("formulas", sa.Column("context_after", sa.Text(), nullable=True))
    op.add_column("formulas", sa.Column("physical_meaning", sa.Text(), nullable=True))
    op.add_column("formulas", sa.Column("semantic_status", sa.String(64), nullable=True))
    op.execute("UPDATE formulas SET context_before = '', context_after = '', semantic_status = 'insufficient_context' WHERE context_before IS NULL")
    op.alter_column("formulas", "context_before", nullable=False)
    op.alter_column("formulas", "context_after", nullable=False)
    op.alter_column("formulas", "semantic_status", nullable=False)


def downgrade() -> None:
    for column in ("semantic_status", "physical_meaning", "context_after", "context_before", "raw_text"):
        op.drop_column("formulas", column)
    for column in ("chunk_index", "formula_ids_json", "content_type", "section_path", "page_end", "page_start"):
        op.drop_column("chunks", column)
