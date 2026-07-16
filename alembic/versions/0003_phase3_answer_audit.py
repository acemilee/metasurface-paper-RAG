"""add phase 3 answer audit fields"""

from alembic import op
import sqlalchemy as sa

revision = "0003_phase3_answer_audit"
down_revision = "0002_phase2_chunks_and_formulas"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("answer_audits", sa.Column("evidence_status", sa.String(64), nullable=True))
    op.add_column("answer_audits", sa.Column("refusal_reason", sa.Text(), nullable=True))
    op.add_column("answer_audits", sa.Column("hallucination_risk", sa.String(32), nullable=True))
    op.add_column("answer_audits", sa.Column("audit_result", sa.String(64), nullable=True))
    op.add_column("answer_audits", sa.Column("citation_ids_json", sa.Text(), nullable=True))
    op.add_column("answer_audits", sa.Column("model_name", sa.String(128), nullable=True))
    op.add_column("answer_audits", sa.Column("prompt_version", sa.String(64), nullable=True))
    op.execute(
        "UPDATE answer_audits SET evidence_status='insufficient', "
        "hallucination_risk='unknown', audit_result='not_run', "
        "citation_ids_json='[]', model_name='deepseek-v4-flash', "
        "prompt_version='grounded-answer-v1'"
    )
    for column in (
        "evidence_status",
        "hallucination_risk",
        "audit_result",
        "citation_ids_json",
        "model_name",
        "prompt_version",
    ):
        op.alter_column("answer_audits", column, nullable=False)


def downgrade() -> None:
    for column in (
        "prompt_version",
        "model_name",
        "citation_ids_json",
        "audit_result",
        "hallucination_risk",
        "refusal_reason",
        "evidence_status",
    ):
        op.drop_column("answer_audits", column)
