from __future__ import annotations

from enum import StrEnum

from paper_rag.config import Settings
from paper_rag.schemas.query_plan import AnswerMode


class DeepSeekTask(StrEnum):
    REWRITE = "rewrite"
    SCHEMA_REPAIR = "schema_repair"
    GENERATION = "generation"
    AUDIT = "audit"
    HYPOTHESIS = "hypothesis"


def thinking_extra_body(
    settings: Settings,
    task: DeepSeekTask,
    answer_mode: AnswerMode | None = None,
) -> dict[str, dict[str, str]]:
    if task == DeepSeekTask.REWRITE:
        enabled = getattr(settings, "deepseek_thinking_rewrite", False)
    elif task == DeepSeekTask.SCHEMA_REPAIR:
        enabled = getattr(settings, "deepseek_thinking_schema_repair", False)
    elif task == DeepSeekTask.GENERATION and answer_mode == AnswerMode.EXTRACT:
        enabled = getattr(settings, "deepseek_thinking_extract", False)
    elif task == DeepSeekTask.GENERATION:
        enabled = getattr(settings, "deepseek_thinking_synthesis", True)
    elif task == DeepSeekTask.AUDIT:
        enabled = getattr(settings, "deepseek_thinking_audit", True)
    else:
        enabled = getattr(settings, "deepseek_thinking_hypothesis", True)
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}
