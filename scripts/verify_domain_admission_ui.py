from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Route, sync_playwright


BASE_URL = "http://127.0.0.1:8010"
RESULT_PATH = Path("evals/results/domain-admission-ui-acceptance.json")
SCREENSHOT_PATH = Path("evals/results/domain-admission-ui.png")
REVIEW_ID = "20000000-0000-0000-0000-000000000001"
DEPENDENCY_ID = "20000000-0000-0000-0000-000000000002"
QUARANTINE_ID = "20000000-0000-0000-0000-000000000003"
ASSESSMENT_ID = "30000000-0000-0000-0000-000000000001"
DEPENDENCY_ASSESSMENT_ID = "30000000-0000-0000-0000-000000000002"
CONVERSATION_ID = "40000000-0000-0000-0000-000000000001"


def _document(
    document_id: str,
    filename: str,
    domain_status: str,
    decision_code: str,
    assessment_id: str,
) -> dict:
    failed = (
        ["embedding_provider"]
        if decision_code == "gate_dependency_unavailable"
        else ["domain_relationship"]
    )
    return {
        "document_id": document_id,
        "original_filename": filename,
        "status": domain_status,
        "pdf_type": "text_pdf",
        "page_count": 12,
        "chunk_count": 0,
        "domain_status": domain_status,
        "domain_score": None,
        "domain_reasons": failed,
        "domain_assessment_id": assessment_id,
        "domain_decision_code": decision_code,
        "domain_passed_requirements": ["parse_quality", "semantic_support"],
        "domain_failed_requirements": failed,
        "domain_evidence": [
            {
                "page_numbers": [2],
                "excerpt": "metasurface mentioned without a supported relationship",
            }
        ],
        "document_genre": "unclassified",
        "genre_score": None,
        "genre_decision_source": None,
        "genre_scores": {},
        "genre_evidence": [],
        "genre_conflicts": [],
        "genre_manually_overridden": False,
        "profile_status": None,
        "created_at": "2026-07-15T12:00:00+08:00",
    }


def main() -> None:
    approve_payloads: list[dict] = []
    reindex_calls: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    conversation_created = False

    documents = [
        _document(
            REVIEW_ID,
            "needs-domain-review.pdf",
            "review_required",
            "missing_domain_relationship",
            ASSESSMENT_ID,
        ),
        _document(
            DEPENDENCY_ID,
            "embedding-temporarily-unavailable.pdf",
            "review_required",
            "gate_dependency_unavailable",
            DEPENDENCY_ASSESSMENT_ID,
        ),
        _document(
            QUARANTINE_ID,
            "unsafe-file.pdf",
            "quarantined",
            "file_security_rejected",
            "30000000-0000-0000-0000-000000000003",
        ),
    ]

    def api_route(route: Route) -> None:
        nonlocal conversation_created
        request = route.request
        path = urlparse(request.url).path
        if path == "/api/documents" and request.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"items": documents, "next_cursor": None}),
            )
            return
        conversation_summary = {
            "conversation_id": CONVERSATION_ID,
            "title": "领域准入验收",
            "scope": "all",
            "document_ids": [],
            "message_count": 0,
            "created_at": "2026-07-15T12:00:00+08:00",
            "updated_at": "2026-07-15T12:00:00+08:00",
        }
        if path == "/api/conversations" and request.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {"items": [conversation_summary] if conversation_created else []}
                ),
            )
            return
        if path == "/api/conversations" and request.method == "POST":
            conversation_created = True
            route.fulfill(
                status=201,
                content_type="application/json",
                body=json.dumps(conversation_summary),
            )
            return
        if path == f"/api/conversations/{CONVERSATION_ID}":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {**conversation_summary, "messages": [], "summary": {}}
                ),
            )
            return
        if path.endswith("/approve"):
            approve_payloads.append(request.post_data_json or {})
            route.fulfill(
                status=202,
                content_type="application/json",
                body=json.dumps(
                    {
                        "document_id": REVIEW_ID,
                        "job_id": "50000000-0000-0000-0000-000000000001",
                        "status": "queued",
                        "assessment_id": ASSESSMENT_ID,
                        "override_id": "60000000-0000-0000-0000-000000000001",
                    }
                ),
            )
            return
        if path.endswith("/reindex"):
            reindex_calls.append(path)
            route.fulfill(
                status=202,
                content_type="application/json",
                body=json.dumps(
                    {
                        "document_id": DEPENDENCY_ID,
                        "job_id": "50000000-0000-0000-0000-000000000002",
                        "duplicate": True,
                        "status": "queued",
                    }
                ),
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            document_id = REVIEW_ID if job_id.endswith("1") else DEPENDENCY_ID
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "job_id": job_id,
                        "document_id": document_id,
                        "state": "completed",
                        "error_code": None,
                        "error_message": None,
                    }
                ),
            )
            return
        if path.endswith("/deletion-check"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "document_id": REVIEW_ID,
                        "original_filename": "needs-domain-review.pdf",
                        "stored_domain_status": "review_required",
                        "fresh_assessment_id": ASSESSMENT_ID,
                        "fresh_domain_status": "review_required",
                        "fresh_decision_code": "missing_domain_relationship",
                        "passed_requirements": ["parse_quality"],
                        "failed_requirements": ["domain_relationship"],
                        "evidence": [{"page_numbers": [2], "excerpt": "evidence"}],
                        "page_count": 12,
                        "chunk_count": 0,
                        "vector_count": 0,
                        "answer_audit_count": 0,
                        "warning": None,
                        "confirmation_token": "token",
                        "expires_in_seconds": 300,
                    }
                ),
            )
            return
        route.fulfill(
            status=404,
            content_type="application/json",
            body=json.dumps({"detail": f"unhandled test route: {path}"}),
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 820})
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.route("**/api/**", api_route)
        page.goto(BASE_URL, wait_until="networkidle")
        page.get_by_text("needs-domain-review.pdf").wait_for()

        assert page.get_by_text("正向证据不足，尚未进入知识索引").count() == 1
        assert page.get_by_text("缺少领域对象与电磁作用关系").count() >= 1
        assert page.get_by_text("相关度差值").count() == 0
        assert page.locator(f'[data-approve-id="{QUARANTINE_ID}"]').count() == 0

        page.locator(f'[data-approve-id="{REVIEW_ID}"]').click()
        page.locator(f'[data-reindex-id="{DEPENDENCY_ID}"]').click()
        assert approve_payloads == [{"assessment_id": ASSESSMENT_ID}]
        assert reindex_calls == [f"/api/documents/{DEPENDENCY_ID}/reindex"]

        page.locator(f'[data-delete-id="{REVIEW_ID}"]').click()
        page.get_by_text("缺少领域对象与电磁作用关系").last.wait_for()
        assert page.get_by_text("相关度差值").count() == 0
        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()

    result = {
        "status": "PASS",
        "approve_payloads": approve_payloads,
        "reindex_calls": reindex_calls,
        "console_errors": console_errors,
        "page_errors": page_errors,
    }
    if console_errors or page_errors:
        result["status"] = "FAIL"
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["status"] == "PASS", result
    print("domain admission UI acceptance: PASS")


if __name__ == "__main__":
    main()
