"""add versioned paper profiles"""

from alembic import op
import sqlalchemy as sa


revision = "0012_paper_profiles"
down_revision = "0011_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("content_json", sa.Text(), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "profile_version", name="uq_paper_profile_document_version"),
    )
    op.create_index("ix_paper_profiles_document_id", "paper_profiles", ["document_id"])
    op.create_index("ix_paper_profiles_source_sha256", "paper_profiles", ["source_sha256"])
    op.create_index("ix_paper_profiles_status", "paper_profiles", ["status"])
    op.create_table(
        "paper_profile_claims",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("claim_type", sa.String(length=64), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("citation_ids_json", sa.Text(), nullable=False),
        sa.Column("audit_verdict", sa.String(length=64), nullable=False),
        sa.Column("evidence_roles_json", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["paper_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_paper_profile_claims_claim_type", "paper_profile_claims", ["claim_type"])
    op.create_index("ix_paper_profile_claims_profile_id", "paper_profile_claims", ["profile_id"])
    op.create_table(
        "paper_profile_relations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("source_entity", sa.String(length=512), nullable=False),
        sa.Column("relation", sa.String(length=128), nullable=False),
        sa.Column("target_entity", sa.String(length=512), nullable=False),
        sa.Column("conditions_json", sa.Text(), nullable=False),
        sa.Column("citation_ids_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["paper_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_paper_profile_relations_profile_id", "paper_profile_relations", ["profile_id"])


def downgrade() -> None:
    op.drop_index("ix_paper_profile_relations_profile_id", table_name="paper_profile_relations")
    op.drop_table("paper_profile_relations")
    op.drop_index("ix_paper_profile_claims_profile_id", table_name="paper_profile_claims")
    op.drop_index("ix_paper_profile_claims_claim_type", table_name="paper_profile_claims")
    op.drop_table("paper_profile_claims")
    op.drop_index("ix_paper_profiles_status", table_name="paper_profiles")
    op.drop_index("ix_paper_profiles_source_sha256", table_name="paper_profiles")
    op.drop_index("ix_paper_profiles_document_id", table_name="paper_profiles")
    op.drop_table("paper_profiles")
