from paper_rag.services.ocr import should_use_ocr


def test_ocr_is_only_used_for_low_text_pages() -> None:
    assert should_use_ocr("short", 1, min_page_chars=80)
    assert should_use_ocr("", 0, min_page_chars=80)
    assert not should_use_ocr("A" * 100, 2, min_page_chars=80)
