from pathlib import Path


def sample_pdf_path() -> Path:
    matches = list(Path.cwd().glob("Dynamical absorption manipulation in a graphene-based optically transparent and flexible metasurface.pdf"))
    if not matches:
        raise FileNotFoundError("Fixed Phase 1 sample PDF is missing")
    return matches[0]
