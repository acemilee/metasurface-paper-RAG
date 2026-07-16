from fastapi.testclient import TestClient

from paper_rag.main import create_app


def test_health_returns_ok() -> None:
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_upload_interface() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert 'id="upload-form"' in response.text


def test_root_exposes_multi_pdf_queue_controls() -> None:
    response = TestClient(create_app()).get("/")

    assert 'id="pdf-file" type="file" accept="application/pdf,.pdf" multiple' in response.text
    assert 'id="upload-queue"' in response.text
    assert 'id="upload-items"' in response.text
    assert 'id="cancel-upload"' in response.text


def test_frontend_contract_uses_independent_scroll_and_batch_polling() -> None:
    client = TestClient(create_app())
    css = client.get("/static/app.css").text
    javascript = client.get("/static/app.js").text

    assert "height: 100dvh" in css
    assert "body {" in css and "overflow: hidden" in css
    assert ".document-list" in css and "overflow-y: auto" in css
    assert ".conversation" in css and "overflow-y: auto" in css
    assert "async function uploadFilesSequentially" in javascript
    assert 'fetch("/api/jobs/batch"' in javascript
    assert "elements.conversation.scrollTo" in javascript
