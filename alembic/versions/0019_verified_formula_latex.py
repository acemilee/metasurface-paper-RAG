"""add verified formula latex and source crop audit fields"""

from alembic import op
import sqlalchemy as sa


revision = "0019_verified_formula_latex"
down_revision = "0018_formula_dependencies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("formulas", sa.Column("latex_text", sa.Text(), nullable=True))
    op.add_column(
        "formulas",
        sa.Column("latex_verification_status", sa.String(32), nullable=False, server_default="absent"),
    )
    op.add_column(
        "formulas",
        sa.Column("latex_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "formulas",
        sa.Column("source_crop_sha256", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("formulas", "source_crop_sha256")
    op.drop_column("formulas", "latex_verified_at")
    op.drop_column("formulas", "latex_verification_status")
    op.drop_column("formulas", "latex_text")
