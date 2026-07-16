"""add positive domain admission audit"""

from alembic import op
import sqlalchemy as sa


revision = "0021_domain_positive_admission"
down_revision = "0020_filename_search_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("domain_enforcement_version", sa.String(64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("domain_decision_code", sa.String(64), nullable=True),
    )
    op.create_table(
        "domain_assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("trigger", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("decision_code", sa.String(64), nullable=False),
        sa.Column("classifier_version", sa.String(64), nullable=False),
        sa.Column("embedding_model_id", sa.String(255), nullable=False),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=False),
        sa.Column("passed_requirements_json", sa.Text(), nullable=False),
        sa.Column("failed_requirements_json", sa.Text(), nullable=False),
        sa.Column("parse_quality", sa.Float(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("applied_to_document", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_domain_assessments_document_id",
        "domain_assessments",
        ["document_id"],
    )
    op.create_index(
        "ix_domain_assessments_decision",
        "domain_assessments",
        ["decision"],
    )
    op.create_index(
        "ix_domain_assessments_applied_to_document",
        "domain_assessments",
        ["applied_to_document"],
    )
    op.create_table(
        "domain_manual_overrides",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["assessment_id"],
            ["domain_assessments.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_domain_manual_overrides_document_id",
        "domain_manual_overrides",
        ["document_id"],
    )
    op.create_index(
        "ix_domain_manual_overrides_assessment_id",
        "domain_manual_overrides",
        ["assessment_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_domain_manual_overrides_assessment_id",
        table_name="domain_manual_overrides",
    )
    op.drop_index(
        "ix_domain_manual_overrides_document_id",
        table_name="domain_manual_overrides",
    )
    op.drop_table("domain_manual_overrides")
    op.drop_index(
        "ix_domain_assessments_applied_to_document",
        table_name="domain_assessments",
    )
    op.drop_index(
        "ix_domain_assessments_decision",
        table_name="domain_assessments",
    )
    op.drop_index(
        "ix_domain_assessments_document_id",
        table_name="domain_assessments",
    )
    op.drop_table("domain_assessments")
    op.drop_column("documents", "domain_decision_code")
    op.drop_column("documents", "domain_enforcement_version")
