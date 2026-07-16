from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


BASE_URL = "http://127.0.0.1:8010"
RESULTS_DIR = Path("evals/results")
CONVERSATION_ID = "10000000-0000-0000-0000-000000000002"


def document_payload(index: int) -> dict:
    return {
        "document_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"document-{index}")),
        "original_filename": f"metasurface-paper-{index:03d}.pdf",
        "status": "completed",
        "pdf_type": "text_pdf",
        "page_count": 9,
        "chunk_count": 12,
        "domain_status": "accepted",
        "domain_score": 0.8,
        "domain_reasons": [],
        "document_genre": "research_paper",
        "genre_score": 0.9,
        "genre_decision_source": "test",
        "genre_scores": {"research_paper": 0.9},
        "genre_evidence": [],
        "genre_conflicts": [],
        "genre_manually_overridden": False,
        "created_at": "2026-07-13T12:00:00+08:00",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    documents = [document_payload(index) for index in range(1, 101)]
    upload_calls: list[str] = []
    accepted_jobs: dict[str, str] = {}
    upload_in_flight = 0
    max_upload_in_flight = 0
    console_errors: list[str] = []

    def api_route(route: Route) -> None:
        nonlocal upload_in_flight, max_upload_in_flight
        request = route.request
        if request.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"items": documents, "next_cursor": None}),
            )
            return
        upload_in_flight += 1
        max_upload_in_flight = max(max_upload_in_flight, upload_in_flight)
        current = len(upload_calls) + 1
        upload_calls.append(f"upload-{current}")
        time.sleep(0.015)
        if current == 7:
            route.fulfill(status=500, content_type="text/plain", body="simulated upload failure")
        else:
            document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"uploaded-document-{current}"))
            job_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"uploaded-job-{current}"))
            accepted_jobs[job_id] = document_id
            route.fulfill(
                status=202,
                content_type="application/json",
                body=json.dumps(
                    {
                        "document_id": document_id,
                        "job_id": job_id,
                        "duplicate": False,
                        "status": "queued",
                    }
                ),
            )
        upload_in_flight -= 1

    def jobs_route(route: Route) -> None:
        payload = route.request.post_data_json
        job_ids = payload.get("job_ids", [])
        jobs = [
            {
                "job_id": job_id,
                "document_id": accepted_jobs[job_id],
                "state": "completed",
                "error_code": None,
                "error_message": None,
            }
            for job_id in job_ids
            if job_id in accepted_jobs
        ]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "jobs": jobs,
                    "missing_job_ids": [job_id for job_id in job_ids if job_id not in accepted_jobs],
                }
            ),
        )

    def conversations_route(route: Route) -> None:
        summary = {
            "conversation_id": CONVERSATION_ID,
            "title": "批量上传布局验收",
            "scope": "all",
            "document_ids": [],
            "message_count": 0,
            "created_at": "2026-07-15T12:00:00+08:00",
            "updated_at": "2026-07-15T12:00:00+08:00",
        }
        payload = (
            {"items": [summary]}
            if route.request.url.rstrip("/").endswith("/api/conversations")
            else {**summary, "messages": [], "summary": {}}
        )
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload, ensure_ascii=False),
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.route("**/api/documents*", api_route)
        page.route("**/api/jobs/batch", jobs_route)
        page.route("**/api/conversations*", conversations_route)
        page.route("**/api/conversations/**", conversations_route)
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        files = [
            {
                "name": f"batch-paper-{index:02d}.pdf",
                "mimeType": "application/pdf",
                "buffer": b"%PDF-1.7\nmock",
            }
            for index in range(1, 41)
        ]
        page.locator("#pdf-file").set_input_files(files)
        page.locator("#upload-submit").click()
        page.wait_for_function("document.querySelectorAll('.upload-item').length === 40")
        page.wait_for_function(
            """() => {
              const statuses = [...document.querySelectorAll('.upload-item-status')].map((item) => item.textContent);
              return statuses.length === 40 && statuses.every((status) => !['等待上传', '上传中'].includes(status));
            }""",
            timeout=30_000,
        )
        page.wait_for_function(
            "document.querySelector('#upload-summary').textContent.includes('0 处理中')",
            timeout=30_000,
        )

        composer_before = page.locator(".composer").bounding_box()
        page.locator("#document-list").evaluate("element => { element.scrollTop = element.scrollHeight; }")
        composer_after = page.locator(".composer").bounding_box()
        desktop_metrics = page.evaluate(
            """() => ({
              bodyScrollY: window.scrollY,
              bodyClientHeight: document.body.clientHeight,
              bodyScrollHeight: document.body.scrollHeight,
              libraryScrollTop: document.querySelector('#document-list').scrollTop,
              libraryScrollable: document.querySelector('#document-list').scrollHeight > document.querySelector('#document-list').clientHeight,
              conversationScrollable: document.querySelector('#conversation').scrollHeight > document.querySelector('#conversation').clientHeight,
              composerBottom: document.querySelector('.composer').getBoundingClientRect().bottom,
              viewportHeight: window.innerHeight
            })"""
        )
        page.screenshot(path=str(RESULTS_DIR / "batch-upload-desktop.png"), full_page=False)

        desktop_viewports = {}
        for width, height in ((1366, 768), (1920, 1080), (2560, 1440)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(100)
            desktop_viewports[f"{width}x{height}"] = page.evaluate(
                """() => ({
                  bodyScrollY: window.scrollY,
                  bodyScrollHeight: document.body.scrollHeight,
                  bodyClientHeight: document.body.clientHeight,
                  composerTop: document.querySelector('.composer').getBoundingClientRect().top,
                  composerBottom: document.querySelector('.composer').getBoundingClientRect().bottom,
                  viewportHeight: window.innerHeight
                })"""
            )

        page.locator("#conversation").evaluate(
            "element => { element.innerHTML = '<p>Long answer line</p>'.repeat(120); element.scrollTop = element.scrollHeight; }"
        )
        long_answer_metrics = page.evaluate(
            """() => ({
              bodyScrollY: window.scrollY,
              bodyScrollHeight: document.body.scrollHeight,
              bodyClientHeight: document.body.clientHeight,
              conversationScrollable: document.querySelector('#conversation').scrollHeight > document.querySelector('#conversation').clientHeight,
              conversationScrollTop: document.querySelector('#conversation').scrollTop,
              composerBottom: document.querySelector('.composer').getBoundingClientRect().bottom
            })"""
        )

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)
        mobile_metrics = page.evaluate(
            """() => ({
              bodyScrollY: window.scrollY,
              bodyScrollHeight: document.body.scrollHeight,
              bodyClientHeight: document.body.clientHeight,
              composerTop: document.querySelector('.composer').getBoundingClientRect().top,
              composerBottom: document.querySelector('.composer').getBoundingClientRect().bottom,
              viewportHeight: window.innerHeight
            })"""
        )
        final_summary = page.locator("#upload-summary").text_content()
        page.screenshot(path=str(RESULTS_DIR / "batch-upload-mobile.png"), full_page=False)

        page.reload()
        page.wait_for_load_state("networkidle")
        page.wait_for_function("document.querySelectorAll('.upload-item').length === 39")
        restored_job_count = page.locator(".upload-item").count()
        page.locator("#clear-upload").click()
        capped_files = [
            {
                "name": f"capacity-paper-{index:03d}.pdf",
                "mimeType": "application/pdf",
                "buffer": b"%PDF-1.7\nmock",
            }
            for index in range(1, 102)
        ]
        page.locator("#pdf-file").set_input_files(capped_files)
        page.wait_for_function("document.querySelectorAll('.upload-item').length === 100")
        capped_selection_count = page.locator(".upload-item").count()
        browser.close()

    result = {
        "selected_files": 40,
        "upload_requests": len(upload_calls),
        "max_upload_in_flight": max_upload_in_flight,
        "simulated_failures": 1,
        "accepted_jobs": len(accepted_jobs),
        "final_summary": final_summary,
        "restored_job_count": restored_job_count,
        "capped_selection_count": capped_selection_count,
        "desktop_metrics": desktop_metrics,
        "desktop_viewports": desktop_viewports,
        "long_answer_metrics": long_answer_metrics,
        "mobile_metrics": mobile_metrics,
        "composer_y_unchanged_after_library_scroll": composer_before == composer_after,
        "console_errors": console_errors,
    }
    (RESULTS_DIR / "batch-upload-ui-acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert len(upload_calls) == 40
    assert max_upload_in_flight == 1
    assert len(accepted_jobs) == 39
    assert restored_job_count == 39
    assert capped_selection_count == 100
    assert composer_before == composer_after
    assert desktop_metrics["bodyScrollY"] == 0
    assert desktop_metrics["bodyScrollHeight"] == desktop_metrics["bodyClientHeight"]
    assert desktop_metrics["libraryScrollable"]
    assert not desktop_metrics["conversationScrollable"]
    assert desktop_metrics["composerBottom"] <= desktop_metrics["viewportHeight"]
    assert long_answer_metrics["bodyScrollHeight"] == long_answer_metrics["bodyClientHeight"]
    assert long_answer_metrics["conversationScrollable"]
    for metrics in desktop_viewports.values():
        assert metrics["bodyScrollY"] == 0
        assert metrics["bodyScrollHeight"] == metrics["bodyClientHeight"]
        assert 0 <= metrics["composerTop"] < metrics["composerBottom"] <= metrics["viewportHeight"]
    assert mobile_metrics["bodyScrollHeight"] == mobile_metrics["bodyClientHeight"]
    assert 0 <= mobile_metrics["composerTop"] < mobile_metrics["composerBottom"] <= mobile_metrics["viewportHeight"]
    unexpected_console_errors = [error for error in console_errors if "500 (Internal Server Error)" not in error]
    assert not unexpected_console_errors
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
