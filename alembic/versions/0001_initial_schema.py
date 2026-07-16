"""create phase 1 tables"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    document_status = sa.Enum("QUEUED", "PARSING", "COMPLETED", "FAILED", name="documentstatus", create_type=False)
    job_state = sa.Enum("QUEUED", "CLASSIFYING", "PARSING", "COMPLETED", "FAILED", name="jobstate", create_type=False)
    op.create_table("documents", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("original_filename", sa.String(512), nullable=False), sa.Column("stored_path", sa.String(1024), nullable=False, unique=True), sa.Column("file_sha256", sa.String(64), nullable=False, unique=True), sa.Column("page_count", sa.Integer(), nullable=True), sa.Column("pdf_type", sa.String(64), nullable=True), sa.Column("status", document_status, nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False))
    op.create_index("ix_documents_file_sha256", "documents", ["file_sha256"])
    op.create_table("ingestion_jobs", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False), sa.Column("state", job_state, nullable=False), sa.Column("worker_id", sa.String(128), nullable=True), sa.Column("error_code", sa.String(64), nullable=True), sa.Column("error_message", sa.Text(), nullable=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), nullable=True), sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_state", "ingestion_jobs", ["state"])
    op.create_table("pages", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False), sa.Column("page_number", sa.Integer(), nullable=False), sa.Column("text", sa.Text(), nullable=False), sa.UniqueConstraint("document_id", "page_number", name="uq_pages_document_page"))
    op.create_index("ix_pages_document_id", "pages", ["document_id"])
    op.create_table("text_blocks", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("page_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("pages.id", ondelete="CASCADE"), nullable=False), sa.Column("reading_order", sa.Integer(), nullable=False), sa.Column("text", sa.Text(), nullable=False), sa.Column("x0", sa.Float(), nullable=False), sa.Column("y0", sa.Float(), nullable=False), sa.Column("x1", sa.Float(), nullable=False), sa.Column("y1", sa.Float(), nullable=False))
    op.create_index("ix_text_blocks_page_id", "text_blocks", ["page_id"])
    op.create_table("chunks", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False), sa.Column("vector_id", sa.String(255), nullable=False, unique=True), sa.Column("content", sa.Text(), nullable=False))
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_table("formulas", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False), sa.Column("page_number", sa.Integer(), nullable=False), sa.Column("placeholder", sa.String(255), nullable=False, unique=True), sa.Column("bbox_json", sa.Text(), nullable=False))
    op.create_index("ix_formulas_document_id", "formulas", ["document_id"])
    op.create_table("answer_audits", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True), sa.Column("question", sa.Text(), nullable=False), sa.Column("answer", sa.Text(), nullable=False))


def downgrade() -> None:
    for table in ("answer_audits", "formulas", "chunks", "text_blocks", "pages", "ingestion_jobs", "documents"):
        op.drop_table(table)
