from __future__ import annotations

import json
from types import SimpleNamespace

from sqlalchemy import select

from paper_rag.models.conversation import Conversation, ConversationTurn
from paper_rag.schemas.chat import AnswerResponse, ChatRequest
from paper_rag.schemas.query_plan import EntityType
from paper_rag.services.conversation_memory import (
    begin_conversation_turn,
    complete_conversation_turn,
)
from paper_rag.services.references import serialize_reference_resolutions
from tests.unit.reference_test_support import formula_extract_plan, sample_resolved_formula


def test_reference_resolution_is_serialized_without_reasoning_or_secrets() -> None:
    resolution = sample_resolved_formula()
    payload = serialize_reference_resolutions([resolution])
    assert payload[0]["resolution_status"] == "resolved"
    assert payload[0]["target_ids"]
    assert "reasoning" not in json.dumps(payload).lower()
    assert "sk-" not in json.dumps(payload).lower()


def test_conversation_turn_persists_strong_and_soft_entity_links(session) -> None:
    conversation = Conversation(title="Reference audit")
    session.add(conversation)
    session.commit()
    request = ChatRequest(
        session_id="session-reference",
        conversation_id=conversation.id,
        client_turn_id="turn-reference-1",
        question="公式5讲了什么",
    )
    begin_conversation_turn(session, request)
    strong = sample_resolved_formula().as_dict()
    soft = {
        "surface": "石墨烯",
        "canonical": "graphene",
        "entity_type": EntityType.MATERIAL.value,
        "must_link": True,
        "linked": True,
        "matched_document_ids": [],
    }
    response = AnswerResponse(
        answer="grounded",
        citations=[],
        evidence_status="sufficient",
        refused=False,
        refusal_reason=None,
        hallucination_risk="low",
        audit_result="passed",
        entity_links=[strong, soft],
    )
    linked_entity = SimpleNamespace(
        surface="石墨烯",
        canonical="graphene",
        entity_type=EntityType.MATERIAL,
        linked=True,
        matched_document_ids=[],
    )

    complete_conversation_turn(
        session,
        request,
        response,
        formula_extract_plan(entity_surface="公式5"),
        [linked_entity],
    )

    turn = session.scalar(select(ConversationTurn))
    persisted = json.loads(turn.entity_links_json)
    assert {item["entity_type"] for item in persisted} == {"formula", "material"}
    formula_link = next(item for item in persisted if item["entity_type"] == "formula")
    assert formula_link["resolution_status"] == "resolved"
    assert formula_link["target_ids"]
