from paper_rag.models.document import Document
from paper_rag.services.filename_search import normalize_filename_search_key


def test_filename_search_key_normalizes_width_and_case() -> None:
    assert normalize_filename_search_key("Ｇraphene－ＭＳ－１２.PDF") == "graphene-ms-12.pdf"


def test_filename_search_key_preserves_chinese_and_pdf_suffix() -> None:
    assert normalize_filename_search_key("石墨烯超表面．ＰＤＦ") == "石墨烯超表面.pdf"


def test_document_derives_search_key_from_original_filename() -> None:
    document = Document(
        original_filename="Ｋubo－公式.PDF",
        stored_path="data/kubo.pdf",
        file_sha256="a" * 64,
    )
    assert document.filename_search_key == "kubo-公式.pdf"
