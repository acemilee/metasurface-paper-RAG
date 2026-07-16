"""add formula groups and dependency graph"""

from alembic import op
import sqlalchemy as sa


revision = "0018_formula_dependencies"
down_revision = "0017_formula_index_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "formula_groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("group_key", sa.String(128), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False),
        sa.Column("completeness_status", sa.String(32), nullable=False, server_default="complete"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "group_key"),
    )
    op.create_index("ix_formula_groups_document_id", "formula_groups", ["document_id"])
    op.add_column("formulas", sa.Column("formula_group_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_formulas_formula_group_id",
        "formulas",
        "formula_groups",
        ["formula_group_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_formulas_formula_group_id", "formulas", ["formula_group_id"])

    for table_name, columns in (
        (
            "formula_references",
            [
                sa.Column("id", sa.Uuid(), primary_key=True),
                sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
                sa.Column("source_formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False),
                sa.Column("target_formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="SET NULL"), nullable=True),
                sa.Column("referenced_number", sa.String(32), nullable=False),
                sa.Column("source_page", sa.Integer(), nullable=False),
                sa.Column("evidence_text", sa.Text(), nullable=False),
                sa.Column("resolution_status", sa.String(32), nullable=False),
            ],
        ),
        (
            "formula_variable_definitions",
            [
                sa.Column("id", sa.Uuid(), primary_key=True),
                sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
                sa.Column("formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False),
                sa.Column("symbol", sa.String(64), nullable=False),
                sa.Column("definition", sa.Text(), nullable=False),
                sa.Column("source_page", sa.Integer(), nullable=False),
                sa.Column("evidence_text", sa.Text(), nullable=False),
            ],
        ),
        (
            "formula_approximation_conditions",
            [
                sa.Column("id", sa.Uuid(), primary_key=True),
                sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
                sa.Column("formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False),
                sa.Column("condition_text", sa.Text(), nullable=False),
                sa.Column("source_page", sa.Integer(), nullable=False),
                sa.Column("evidence_text", sa.Text(), nullable=False),
            ],
        ),
        (
            "formula_derivation_edges",
            [
                sa.Column("id", sa.Uuid(), primary_key=True),
                sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
                sa.Column("source_formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="SET NULL"), nullable=True),
                sa.Column("target_formula_id", sa.Uuid(), sa.ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False),
                sa.Column("evidence_text", sa.Text(), nullable=False),
                sa.Column("resolution_status", sa.String(32), nullable=False),
            ],
        ),
    ):
        op.create_table(table_name, *columns)
        op.create_index(f"ix_{table_name}_document_id", table_name, ["document_id"])

    op.create_index("ix_formula_references_source_formula_id", "formula_references", ["source_formula_id"])
    op.create_index("ix_formula_references_target_formula_id", "formula_references", ["target_formula_id"])
    op.create_index("ix_formula_variable_definitions_formula_id", "formula_variable_definitions", ["formula_id"])
    op.create_index("ix_formula_approximation_conditions_formula_id", "formula_approximation_conditions", ["formula_id"])
    op.create_index("ix_formula_derivation_edges_source_formula_id", "formula_derivation_edges", ["source_formula_id"])
    op.create_index("ix_formula_derivation_edges_target_formula_id", "formula_derivation_edges", ["target_formula_id"])


def downgrade() -> None:
    for table_name in (
        "formula_derivation_edges",
        "formula_approximation_conditions",
        "formula_variable_definitions",
        "formula_references",
    ):
        op.drop_table(table_name)
    op.drop_index("ix_formulas_formula_group_id", table_name="formulas")
    op.drop_constraint("fk_formulas_formula_group_id", "formulas", type_="foreignkey")
    op.drop_column("formulas", "formula_group_id")
    op.drop_index("ix_formula_groups_document_id", table_name="formula_groups")
    op.drop_table("formula_groups")
