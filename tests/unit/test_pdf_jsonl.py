from uuid import uuid4

from paper_rag.services.pdf_parser import parse_text_pdf, write_page_jsonl
from tests.fixtures.sample_paper import sample_pdf_path


def test_write_page_jsonl_preserves_all_sample_pages(tmp_path) -> None:
    parsed = parse_text_pdf(sample_pdf_path(), uuid4())
    output = tmp_path / "parsed.jsonl"

    write_page_jsonl(parsed, output)

    assert len(output.read_text(encoding="utf-8").splitlines()) == 9
