from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Route, sync_playwright


BASE_URL = "http://127.0.0.1:8010"
RESULT_PATH = Path("evals/results/filename-search-ui-acceptance.json")
DOCUMENTS = [
    ("00000000-0000-0000-0000-000000000001", "Alpha.PDF"),
    ("00000000-0000-0000-0000-000000000002", "石墨烯－１２.pdf"),
    ("00000000-0000-0000-0000-000000000003", "unrelated.pdf"),
]
CONVERSATION_ID = "10000000-0000-0000-0000-000000000001"


def _key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _document(document_id: str, filename: str) -> dict:
    return {
        "document_id": document_id,
        "original_filename": filename,
        "status": "completed",
        "pdf_type": "text_pdf",
        "page_count": 9,
        "chunk_count": 12,
        "domain_status": "accepted",
        "domain_score": 0.9,
        "domain_reasons": [],
        "document_genre": "research_paper",
        "genre_score": 0.9,
        "genre_decision_source": "test",
        "genre_scores": {"research_paper": 0.9},
        "genre_evidence": [],
        "genre_conflicts": [],
        "genre_manually_overridden": False,
        "profile_status": "ready",
        "created_at": "2026-07-15T12:00:00+08:00",
    }


def main() -> None:
    query_calls: list[str] = []
    console_errors: list[str] = []
    http_errors: list[str] = []

    def documents_route(route: Route) -> None:
        request = route.request
        if request.method != "GET":
            route.continue_()
            return
        query = parse_qs(urlparse(request.url).query).get("filename", [""])[0]
        query_calls.append(query)
        if _key(query) == "alpha":
            time.sleep(0.8)
        normalized = _key(query.strip())
        matches = [
            _document(document_id, filename)
            for document_id, filename in DOCUMENTS
            if not normalized or normalized in _key(filename)
        ]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"items": matches, "next_cursor": None}, ensure_ascii=False),
        )

    def conversations_route(route: Route) -> None:
        summary = {
            "conversation_id": CONVERSATION_ID,
            "title": "文件名检索验收",
            "scope": "all",
            "document_ids": [],
            "message_count": 0,
            "created_at": "2026-07-15T12:00:00+08:00",
            "updated_at": "2026-07-15T12:00:00+08:00",
        }
        path = urlparse(route.request.url).path.rstrip("/")
        payload = {"items": [summary]} if path == "/api/conversations" else {
            **summary,
            "messages": [],
            "summary": {},
        }
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload, ensure_ascii=False),
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 420})
        page = context.new_page()
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on(
            "response",
            lambda response: http_errors.append(f"{response.status} {response.url}")
            if response.status >= 400
            else None,
        )
        page.route("**/api/documents*", documents_route)
        page.route("**/api/conversations*", conversations_route)
        page.route("**/api/conversations/**", conversations_route)
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("document.querySelectorAll('.document-item').length === 3")

        alpha = page.locator(
            '[data-document-id="00000000-0000-0000-0000-000000000001"]'
        )
        alpha.check()

        page.locator("#filename-search").fill("石墨烯-12.PDF")
        page.locator("#filename-search").press("Enter")
        page.wait_for_function(
            "document.querySelectorAll('.document-item').length === 2", timeout=150
        )
        assert page.locator(".document-name").all_text_contents() == [
            "石墨烯－１２.pdf",
            "Alpha.PDF",
        ]
        assert page.locator("[data-selected-papers-anchor]").inner_text() == "已选论文 · 1"
        assert page.locator(".document-item").last.locator("input[type=checkbox]").is_checked()
        page.wait_for_function(
            "document.querySelector('#document-list').scrollTop > 0", timeout=500
        )
        page.wait_for_function(
            """() => {
              const list = document.querySelector('#document-list');
              return list.scrollHeight - list.clientHeight - list.scrollTop < 2;
            }""",
            timeout=1_000,
        )
        scroll_metrics = page.evaluate(
            """() => ({
              listScrollTop: document.querySelector('#document-list').scrollTop,
              bodyScrollY: window.scrollY
            })"""
        )
        assert scroll_metrics["listScrollTop"] > 0
        assert scroll_metrics["bodyScrollY"] == 0
        assert page.locator("#library-search-status").is_hidden()
        assert page.locator("#document-count-label").inner_text() == "篇匹配"

        page.locator("#filename-search").fill("missing")
        page.wait_for_timeout(350)
        page.wait_for_function(
            "document.querySelector('#document-list').textContent.includes('未找到匹配的论文文件')"
        )
        assert page.locator(".document-name").all_text_contents() == ["Alpha.PDF"]
        assert page.locator("[data-selected-papers-anchor]").inner_text() == "已选论文 · 1"
        assert page.locator('.document-name:text-is("Alpha.PDF")').count() == 1

        page.locator("#filename-search").fill("")
        page.wait_for_function("document.querySelectorAll('.document-item').length === 3")
        assert alpha.is_checked()

        page.locator("#filename-search").fill("alpha")
        page.locator("#filename-search").press("Enter")
        page.wait_for_function("document.querySelector('#document-count').textContent === '1'")
        assert "未找到匹配的论文文件" not in page.locator("#document-list").inner_text()
        assert page.locator(".document-name").all_text_contents() == ["Alpha.PDF"]
        assert page.locator('.document-name:text-is("Alpha.PDF")').count() == 1
        alpha_checkbox = page.locator('.document-name:text-is("Alpha.PDF")').locator("xpath=ancestor::div[contains(@class, 'document-item')]").locator("input[type=checkbox]")
        alpha_checkbox.uncheck()
        assert page.locator("[data-selected-papers-anchor]").count() == 0
        assert page.locator('.document-name:text-is("Alpha.PDF")').count() == 1
        assert not page.locator('.document-name:text-is("Alpha.PDF")').locator("xpath=ancestor::div[contains(@class, 'document-item')]").locator("input[type=checkbox]").is_checked()
        page.locator('.document-name:text-is("Alpha.PDF")').locator("xpath=ancestor::div[contains(@class, 'document-item')]").locator("input[type=checkbox]").check()
        assert page.locator("[data-selected-papers-anchor]").inner_text() == "已选论文 · 1"
        snapshots = page.evaluate("JSON.parse(sessionStorage.getItem('paper-rag-selected-document-snapshots') || '[]')")
        assert [item["original_filename"] for item in snapshots] == ["Alpha.PDF"]

        page.locator("#filename-search").fill("")
        page.wait_for_function("document.querySelectorAll('.document-item').length === 3")

        page.locator("#filename-search").fill("alpha")
        page.evaluate(
            """setTimeout(() => {
              const input = document.querySelector('#filename-search');
              input.value = '石墨烯';
              input.dispatchEvent(new Event('input', { bubbles: true }));
            }, 300)"""
        )
        page.wait_for_timeout(1_400)
        assert page.locator(".document-name").all_text_contents() == [
            "石墨烯－１２.pdf",
            "Alpha.PDF",
        ]
        assert "alpha" in query_calls and "石墨烯" in query_calls
        assert not http_errors, http_errors
        assert not console_errors, console_errors
        browser.close()

    report = {
        "status": "PASS",
        "whole_library_server_search": True,
        "empty_state": True,
        "clear_restores_list": True,
        "selection_preserved": True,
        "stale_response_rejected": True,
        "queries": query_calls,
        "console_errors": console_errors,
        "http_errors": http_errors,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
