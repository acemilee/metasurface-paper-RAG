from __future__ import annotations

import ast
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.schemas.query_plan import AnswerMode
from paper_rag.services.thinking import DeepSeekTask, thinking_extra_body


ROOT = Path(__file__).resolve().parents[2]
PROVIDER_FILES = [
    ROOT / "src/paper_rag/services/deepseek.py",
    ROOT / "src/paper_rag/services/query_rewrite.py",
]


def _is_completion_create(call: ast.Call) -> bool:
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "create"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "completions"
    )


def test_every_deepseek_call_explicitly_sets_thinking() -> None:
    missing = []
    call_count = 0
    for path in PROVIDER_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_completion_create(node):
                call_count += 1
                if "extra_body" not in {keyword.arg for keyword in node.keywords}:
                    missing.append(f"{path.name}:{node.lineno}")

    assert call_count >= 13
    assert missing == []


def test_thinking_policy_is_task_specific_and_explicit() -> None:
    settings = Settings()

    assert thinking_extra_body(settings, DeepSeekTask.REWRITE) == {"thinking": {"type": "disabled"}}
    assert thinking_extra_body(settings, DeepSeekTask.SCHEMA_REPAIR) == {"thinking": {"type": "disabled"}}
    assert thinking_extra_body(settings, DeepSeekTask.GENERATION, AnswerMode.EXTRACT) == {"thinking": {"type": "disabled"}}
    assert thinking_extra_body(settings, DeepSeekTask.GENERATION, AnswerMode.SYNTHESIZE) == {"thinking": {"type": "enabled"}}
    assert thinking_extra_body(settings, DeepSeekTask.AUDIT) == {"thinking": {"type": "enabled"}}
    assert thinking_extra_body(settings, DeepSeekTask.HYPOTHESIS) == {"thinking": {"type": "enabled"}}

