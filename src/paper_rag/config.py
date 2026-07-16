from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPER_RAG_")

    postgres_dsn: str = "postgresql+psycopg://paper_rag:paper_rag@127.0.0.1:5433/paper_rag"
    upload_dir: Path = Path("data/uploads")
    parsed_dir: Path = Path("data/parsed")
    chroma_dir: Path = Path("data/chroma")
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pdf_pages: int = 200
    queue_maxsize: int = 32
    chunk_target_chars: int = 1400
    chunk_overlap_chars: int = 180
    embedding_provider: str = "http"
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_model_path: Path = Path("models/BAAI-bge-m3")
    embedding_dimension: int = 1024
    embedding_device: str = "auto"
    embedding_batch_size: int = 4
    embedding_max_seq_length: int = 2048
    embedding_service_url: str = "http://127.0.0.1:8011"
    embedding_service_timeout_seconds: float = 120.0
    embedding_service_max_batch_size: int = 32
    embedding_ingestion_batch_size: int = 8
    embedding_service_max_text_chars: int = 12000
    retrieval_min_score: float = 0.50
    retrieval_score_floor: float = 0.42
    retrieval_lexical_min_terms: int = 3
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = 90.0
    deepseek_key_ttl_seconds: int = 3600
    deepseek_max_retries: int = 2
    deepseek_thinking_rewrite: bool = False
    deepseek_thinking_schema_repair: bool = False
    deepseek_thinking_extract: bool = False
    deepseek_thinking_synthesis: bool = True
    deepseek_thinking_audit: bool = True
    deepseek_thinking_hypothesis: bool = True
    prompt_version: str = "grounded-answer-v1"
    ocr_enabled: bool = True
    ocr_render_dpi: int = 216
    ocr_min_page_chars: int = 80
    ocr_min_confidence: float = 0.65
    ocr_numeric_min_confidence: float = 0.85
    ocr_page_timeout_seconds: float = 120.0
    ocr_max_pages_per_document: int = 50
    ocr_cpu_threads: int = 4
    worker_heartbeat_seconds: float = 5.0
    stale_job_seconds: int = 300
    max_job_attempts: int = 3
    readiness_worker_stale_seconds: int = 30
    readiness_min_disk_free_gb: float = 1.0
    worker_max_memory_percent: float = 90.0
    domain_region_min_chars: int = 240
    domain_region_target_chars: int = 1200
    domain_region_max_count: int = 12
    domain_min_evidence_regions: int = 2
    domain_semantic_support_min: float = 0.48
    domain_parse_quality_min: float = 0.65
    domain_gate_max_retries: int = 2
    domain_gate_safe_mode: Literal["enforce", "review_all"] = "enforce"
    deletion_token_ttl_seconds: int = 300
    rewrite_min_fidelity_score: float = 0.45

    def ensure_directories(self) -> None:
        for directory in (self.upload_dir, self.parsed_dir, self.chroma_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
