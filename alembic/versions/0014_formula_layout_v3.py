"""add versioned formula layout metadata"""

from alembic import op
import sqlalchemy as sa


revision = "0014_formula_layout_v3"
down_revision = "0013_conversation_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("formulas", sa.Column("formula_number", sa.String(32), nullable=True))
    op.add_column("formulas", sa.Column("group_key", sa.String(128), nullable=True))
    op.add_column("formulas", sa.Column("part_index", sa.Integer(), nullable=False, server_default="0"))
    op.add_column(
        "formulas",
        sa.Column("parser_version", sa.String(64), nullable=False, server_default="legacy-v1"),
    )
    op.add_column("formulas", sa.Column("normalized_text", sa.Text(), nullable=True))
    op.add_column(
        "formulas",
        sa.Column("fidelity_status", sa.String(32), nullable=False, server_default="needs_review"),
    )


def downgrade() -> None:
    for column in (
        "fidelity_status",
        "normalized_text",
        "parser_version",
        "part_index",
        "group_key",
        "formula_number",
    ):
        op.drop_column("formulas", column)
