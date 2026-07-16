from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_JS = ROOT / "src/paper_rag/static/app.js"


def test_gui_exposes_persistent_conversation_controls() -> None:
    html = (ROOT / "src/paper_rag/templates/index.html").read_text(encoding="utf-8")

    assert 'id="conversation-select"' in html
    assert 'id="new-conversation"' in html
    assert 'id="rename-conversation"' in html
    assert 'id="reset-conversation"' in html
    assert 'id="delete-conversation"' in html


def test_gui_appends_turns_and_sends_conversation_identifiers() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")

    assert "conversation_id: state.conversationId" in javascript
    assert "client_turn_id: clientTurnId" in javascript
    assert "elements.conversation.appendChild" in javascript
    assert "elements.conversation.innerHTML = `<div class=\"question-block\"" not in javascript


def test_conversation_remains_its_own_scroll_container() -> None:
    css = (ROOT / "src/paper_rag/static/app.css").read_text(encoding="utf-8")

    assert ".conversation {" in css
    assert "overflow-y: auto" in css
    assert ".conversation-turn" in css


def test_formula_assets_render_as_source_images() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    css = (ROOT / "src/paper_rag/static/app.css").read_text(encoding="utf-8")

    assert "payload.formula_assets" in javascript
    assert 'class="formula-source-image"' in javascript
    assert ".formula-source-image" in css


def test_verified_formula_mathml_is_parsed_locally_and_pdf_crop_remains_auditable() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")

    assert "DOMParser" in javascript
    assert "http://www.w3.org/1998/Math/MathML" in javascript
    assert "rendered_mathml" in javascript
    assert "查看原文裁剪" in javascript


def test_gui_has_precise_strong_reference_messages() -> None:
    javascript = APP_JS.read_text(encoding="utf-8")
    assert "strong_reference_not_found" in javascript
    assert "strong_reference_ambiguous" in javascript
    assert "reference_index_inconsistent" in javascript
    assert "reference_resolution" in javascript


def test_gui_exposes_corresponding_source_and_license() -> None:
    html = (ROOT / "src/paper_rag/templates/index.html").read_text(encoding="utf-8")
    css = (ROOT / "src/paper_rag/static/app.css").read_text(encoding="utf-8")

    assert "源代码与许可证" in html
    assert "https://github.com/acemilee/metasurface-paper-RAG/tree/v0.1.0" in html
    assert 'rel="noopener noreferrer"' in html
    assert ".source-license-link" in css
