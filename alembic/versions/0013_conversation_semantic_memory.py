"""add bounded semantic conversation memory"""

from alembic import op
import sqlalchemy as sa


revision = "0013_conversation_memory"
down_revision = "0012_paper_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_turns",
        sa.Column("question_embedding_json", sa.Text(), server_default="[]", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("conversation_turns", "question_embedding_json")
