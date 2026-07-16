from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def _payload(audit_result: str, fidelity_status: str | None) -> dict:
    assets = []
    if fidelity_status:
        assets.append(
            {
                "formula_id": "11111111-1111-1111-1111-111111111111",
                "group_key": "equation-1",
                "formula_number": "1a",
                "page_number": 4,
                "image_url": "/missing-formula-image",
                "normalized_text": "source glyphs",
                "fidelity_status": fidelity_status,
            }
        )
    return {
        "answer": "formula result",
        "citations": [],
        "evidence_status": "sufficient" if assets else "insufficient",
        "refused": not assets,
        "refusal_reason": None if assets else "formula unavailable",
        "hallucination_risk": "low" if assets else "unknown",
        "audit_result": audit_result,
        "action": "answer" if assets else "refuse",
        "answer_mode": "extract",
        "epistemic_level": "source_fact",
        "formula_assets": assets,
        "reasoning_content": "SECRET_REASONING",
    }


def main() -> None:
    app_js = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    testable_prefix = app_js.split('elements.documentList.addEventListener("change"', 1)[0]
    testable_prefix = testable_prefix.replace("crypto.randomUUID()", '"test-session-id"')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.route(
            "http://formula.test/**",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body='<main id="conversation"></main>',
            ),
        )
        page.goto("http://formula.test/")
        page.add_script_tag(content=testable_prefix)
        results: dict[str, bool] = {}
        cases = {
            "source_exact": _payload("formula_source_rendered", "source_exact"),
            "needs_review": _payload("formula_source_rendered", "needs_review"),
            "unavailable": _payload("formula_not_extracted", None),
        }
        expected = {
            "source_exact": "公式已可靠提取",
            "needs_review": "公式已定位，文本待复核",
            "unavailable": "公式无法可靠还原",
        }
        for name, payload in cases.items():
            text = page.evaluate(
                """payload => {
                    const container = document.createElement('section');
                    renderAnswer(container, payload);
                    return container.innerText;
                }""",
                payload,
            )
            results[name] = expected[name] in text
            if name == "unavailable":
                results[name] = results[name] and "服务异常" not in text
            results.setdefault("reasoning_hidden", True)
            results["reasoning_hidden"] = results["reasoning_hidden"] and "SECRET_REASONING" not in text
        browser.close()
    print(json.dumps(results, ensure_ascii=False))
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
