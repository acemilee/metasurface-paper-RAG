from pathlib import Path

import pytest


def sample_pdf_path() -> Path:
    matches = list(Path.cwd().glob("Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"))
    if not matches:
        pytest.skip("private regression PDF is not distributed")
    return matches[0]
