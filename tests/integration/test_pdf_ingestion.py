from uuid import uuid4

from paper_rag.services.pdf_parser import parse_text_pdf
from tests.fixtures.sample_paper import sample_pdf_path


def test_sample_pdf_extracts_nine_pages_and_theoretical_model_heading() -> None:
    parsed = parse_text_pdf(sample_pdf_path(), uuid4())

    assert parsed.page_count == 9
    assert "3. Theoretical model" in parsed.pages[3].text
    assert all(page.blocks for page in parsed.pages)
    assert all(block.text and block.page_number == page.page_number for page in parsed.pages for block in page.blocks)
