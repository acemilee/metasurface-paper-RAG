from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_library_exposes_original_filename_search_control() -> None:
    html = (ROOT / "src/paper_rag/templates/index.html").read_text(encoding="utf-8")
    assert 'id="filename-search"' in html
    assert 'type="search"' in html
    assert 'aria-label="按 PDF 文件名检索"' in html
    assert 'id="library-search-status"' in html
    assert 'aria-live="polite"' in html
    assert 'id="document-count-label"' in html
    assert 'class="library-search-status" aria-live="polite" hidden' in html


def test_filename_search_control_fits_fixed_library_panel() -> None:
    css = (ROOT / "src/paper_rag/static/app.css").read_text(encoding="utf-8")
    assert ".library-search" in css
    assert "min-width: 0" in css
    assert ".library-search-status { min-height:" not in css


def test_successful_search_uses_one_compact_result_count_line() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    assert 'state.libraryQuery ? "篇匹配" : "篇已收录"' in javascript
    assert 'elements.librarySearchStatus.textContent = "";' in javascript
    assert "elements.librarySearchStatus.hidden = true;" in javascript


def test_selected_paper_snapshots_are_session_scoped_and_safe_to_restore() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    assert '"paper-rag-selected-document-snapshots"' in javascript
    assert "restoreSelectedDocumentSnapshots" in javascript
    assert "selectedDocumentSnapshots: restoreSelectedDocumentSnapshots()" in javascript
    assert "state.documentCache" in javascript
    assert "sessionStorage.setItem(selectedSnapshotStorageKey" in javascript


def test_search_view_pins_selected_papers_once_at_the_bottom() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    css = (ROOT / "src/paper_rag/static/app.css").read_text(encoding="utf-8")
    assert "function deriveLibraryView()" in javascript
    assert "matchedDocuments: state.documents.filter" in javascript
    assert "selectedDocuments: [...state.selectedIds]" in javascript
    assert "data-selected-papers-anchor" in javascript
    assert "已选论文 · ${selectedDocuments.length}" in javascript
    assert ".selected-papers-divider" in css


def test_new_search_scrolls_only_the_library_to_selected_papers() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    assert "function scrollSelectedPapersIntoView()" in javascript
    assert "elements.documentList.scrollTo" in javascript
    assert "top: elements.documentList.scrollHeight" in javascript
    assert "window.scrollTo" not in javascript
    assert "loadDocuments({ revealSelected: true })" in javascript


def test_gui_searches_server_and_guards_request_races() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    assert 'url.searchParams.set("filename", state.libraryQuery)' in javascript
    assert "state.libraryRequestController?.abort()" in javascript
    assert "libraryRequestGeneration" in javascript
    assert 'error.name === "AbortError"' in javascript


def test_new_search_input_invalidates_in_flight_response_before_debounce() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    handler = javascript.split(
        'elements.filenameSearch.addEventListener("input", () => {', 1
    )[1].split("});", 1)[0]
    assert "state.libraryRequestController?.abort();" in handler
    assert "state.libraryRequestGeneration += 1;" in handler


def test_enter_submits_filename_search_without_waiting_for_debounce() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    handler = javascript.split(
        'elements.filenameSearch.addEventListener("keydown", (event) => {', 1
    )[1].split("});", 1)[0]
    assert 'event.key !== "Enter"' in handler
    assert "event.preventDefault();" in handler
    assert "window.clearTimeout(librarySearchTimer);" in handler
    assert "loadDocuments({ revealSelected:" in handler


def test_rendering_does_not_prune_selected_documents_to_visible_results() -> None:
    javascript = (ROOT / "src/paper_rag/static/app.js").read_text(encoding="utf-8")
    assert "filter((id) => knownIds.has(id))" not in javascript
    assert "state.selectedIds.delete(check.document_id)" in javascript


def test_project_memory_records_filename_search_boundary() -> None:
    memo = (ROOT / "docs/MVP_UX_AND_LIBRARY_GAPS.md").read_text(encoding="utf-8")
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    boundary = "仅检索 PDF 原始文件名"
    assert boundary in memo
    assert boundary in handoff
    assert "不检索标题、作者、DOI或正文" in memo
