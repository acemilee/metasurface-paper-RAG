"""make the current formula parser the default for new records"""

from alembic import op
import sqlalchemy as sa


revision = "0016_formula_parser_default"
down_revision = "0015_formula_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "formulas",
        "parser_version",
        existing_type=sa.String(64),
        server_default="formula-layout-v3",
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "formulas",
        "parser_version",
        existing_type=sa.String(64),
        server_default="legacy-v1",
        existing_nullable=False,
    )
