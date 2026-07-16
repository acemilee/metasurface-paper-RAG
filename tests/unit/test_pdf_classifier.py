from paper_rag.services.pdf_classifier import PdfType, classify_pdf
from tests.fixtures.sample_paper import sample_pdf_path


def test_sample_pdf_is_digital_text_pdf() -> None:
    result = classify_pdf(sample_pdf_path())

    assert result.pdf_type is PdfType.DIGITAL_TEXT
    assert result.page_count == 9
