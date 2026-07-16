"""add persistent research conversations"""

from alembic import op
import sqlalchemy as sa


revision = "0011_conversations"
down_revision = "0010_document_genre_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("selected_document_ids_json", sa.Text(), nullable=False),
        sa.Column("summary_json", sa.Text(), nullable=False),
        sa.Column("summary_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "turn_index", "role", name="uq_conversation_message_turn_role"),
    )
    op.create_index("ix_conversation_messages_conversation_id", "conversation_messages", ["conversation_id"])
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("client_turn_id", sa.String(length=128), nullable=False),
        sa.Column("original_question", sa.Text(), nullable=False),
        sa.Column("standalone_question", sa.Text(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("selected_document_ids_json", sa.Text(), nullable=False),
        sa.Column("query_plan_json", sa.Text(), nullable=False),
        sa.Column("entity_links_json", sa.Text(), nullable=False),
        sa.Column("citation_ids_json", sa.Text(), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("audit_result", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "client_turn_id", name="uq_conversation_client_turn"),
        sa.UniqueConstraint("conversation_id", "turn_index", name="uq_conversation_turn_index"),
    )
    op.create_index("ix_conversation_turns_conversation_id", "conversation_turns", ["conversation_id"])
    op.create_table(
        "conversation_entities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("canonical", sa.String(length=512), nullable=False),
        sa.Column("surface", sa.String(length=512), nullable=False),
        sa.Column("document_ids_json", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("last_turn_index", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "entity_type", "canonical", name="uq_conversation_entity"),
    )
    op.create_index("ix_conversation_entities_conversation_id", "conversation_entities", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_conversation_entities_conversation_id", table_name="conversation_entities")
    op.drop_table("conversation_entities")
    op.drop_index("ix_conversation_turns_conversation_id", table_name="conversation_turns")
    op.drop_table("conversation_turns")
    op.drop_index("ix_conversation_messages_conversation_id", table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
