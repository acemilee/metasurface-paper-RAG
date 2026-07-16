from __future__ import annotations

import unicodedata


def normalize_filename_search_key(filename: str) -> str:
    return unicodedata.normalize("NFKC", filename).casefold()
