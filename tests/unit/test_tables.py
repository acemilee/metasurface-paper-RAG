from paper_rag.services.tables import TableCandidate, table_to_markdown


def test_table_to_markdown_preserves_cells() -> None:
    candidate = TableCandidate(2, [["Bias", "Band"], ["5 V", "7-18.2 GHz"]], 0.9)

    markdown = table_to_markdown(candidate)

    assert "| Bias | Band |" in markdown
    assert "| 5 V | 7-18.2 GHz |" in markdown
